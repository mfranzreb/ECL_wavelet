/*
BSD 3-Clause License

Copyright (c) 2025, Marco Franzreb, Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <bit_array.cuh>
#include <cstdint>
#include <cub/device/device_scan.cuh>

namespace ecl {
/*!
 * \brief Static configuration for \c RankSelect.
 */
struct RSConfig {
  // Bits covered by an L2-block.
  static constexpr size_t L2_BIT_SIZE = 512;

  // Number of L2-blocks per L1-block
  static constexpr size_t NUM_L2_PER_L1 = 128;

  // Bits covered by an L1-block.
  static constexpr size_t L1_BIT_SIZE = NUM_L2_PER_L1 * L2_BIT_SIZE;

  // Number of 32-bit words covered by an L2-block.
  static constexpr size_t L2_WORD_SIZE = L2_BIT_SIZE / (sizeof(uint32_t) * 8);
  // Number of 32-bit words covered by an L1-block.
  static constexpr size_t L1_WORD_SIZE = L1_BIT_SIZE / (sizeof(uint32_t) * 8);

  // Number of 1s and 0s between each sampled position.
  static constexpr size_t SELECT_SAMPLE_RATE =
      4096;  // TODO: try different sizes

  using L1_TYPE = uint64_t;
  using L2_TYPE = uint16_t;
};  // struct RankSelectConfiguration

struct RankResult {
  size_t rank;
  bool bit;
};

class RankSelect;

/*!
 * \brief Fill L2 indices and prepare L1 indices for prefix sum.
 * \param rank_select RankSelect object to fill indices for.
 * \param array_index Index of the bit array to be used.
 * \param num_last_l2_blocks Number of L2 blocks in the last L1 block.
 */
__global__ static void calculateL2EntriesKernel(
    RankSelect rank_select, uint32_t const array_index,
    uint16_t const num_last_l2_blocks, size_t const num_l1_blocks,
    uint32_t const num_threads, uint32_t const num_iters,
    size_t const bit_array_size);

__global__ static void calculateSelectSamplesKernel(
    RankSelect rank_select, uint32_t const array_index,
    size_t const num_threads, size_t const num_ones_samples,
    size_t const num_zeros_samples);

__global__ static void addNumOnesKernel(RankSelect rank_select,
                                        uint32_t const num_arrays);

/*!
 * \brief Rank and select support for the bit array.
 */
class RankSelect {
 public:
  BitArray bit_array_; /*!< Bitarray the object wraps.*/

  /*!
   * \brief Default constructor.
   */
  __host__ RankSelect() noexcept = default;

  /*!
   * \brief Constructor. Creates the auxiliary information for efficient rank
   * and select queries.
   * \param bit_array \c BitArray to be used for queries.
   * \param GPU_index Index of the GPU to be used.
   */
  __host__ RankSelect(BitArray&& bit_array, uint8_t const GPU_index) noexcept
      : bit_array_(std::move(bit_array)), is_copy_(false) {
    checkWarpSize(GPU_index);

    auto const num_arrays = bit_array_.numArrays();

    // Compute the number of L1 blocks.
    num_l1_blocks_.resize(num_arrays);
    for (size_t i = 0; i < num_arrays; ++i) {
      num_l1_blocks_[i] = (bit_array_.sizeHost(i) + RSConfig::L1_BIT_SIZE - 1) /
                          RSConfig::L1_BIT_SIZE;
    }
    // transfer to device
    gpuErrchk(cudaMallocAsync(&d_num_l1_blocks_,
                              num_l1_blocks_.size() * sizeof(size_t),
                              cudaStreamDefault));
    gpuErrchk(cudaMemcpyAsync(d_num_l1_blocks_, num_l1_blocks_.data(),
                              num_l1_blocks_.size() * sizeof(size_t),
                              cudaMemcpyHostToDevice, cudaStreamDefault));
    size_t total_l1_blocks = 0;
    for (auto const& num_blocks : num_l1_blocks_) {
      total_l1_blocks += num_blocks;
    }
    // Allocate memory for the L1 index.
    // For convenience, first entry is for the first block, which is
    // always 0.
    gpuErrchk(cudaMallocAsync(&d_l1_indices_,
                              total_l1_blocks * sizeof(RSConfig::L1_TYPE),
                              cudaStreamDefault));
    gpuErrchk(cudaMemsetAsync(d_l1_indices_, 0,
                              total_l1_blocks * sizeof(RSConfig::L1_TYPE),
                              cudaStreamDefault));

    std::vector<size_t> l1_offsets(num_l1_blocks_.size());

    std::exclusive_scan(num_l1_blocks_.begin(), num_l1_blocks_.end(),
                        l1_offsets.begin(), 0);
    gpuErrchk(cudaMallocAsync(
        &d_l1_offsets_, l1_offsets.size() * sizeof(size_t), cudaStreamDefault));
    gpuErrchk(cudaMemcpyAsync(d_l1_offsets_, l1_offsets.data(),
                              l1_offsets.size() * sizeof(size_t),
                              cudaMemcpyHostToDevice, cudaStreamDefault));

    // Get how many l2 blocks each last L1 block has
    std::vector<uint16_t> num_last_l2_blocks(num_arrays);
    for (size_t i = 0; i < num_arrays; ++i) {
      num_last_l2_blocks[i] = (bit_array_.sizeHost(i) % RSConfig::L1_BIT_SIZE +
                               RSConfig::L2_BIT_SIZE - 1) /
                              RSConfig::L2_BIT_SIZE;
      if (num_last_l2_blocks[i] == 0) {
        num_last_l2_blocks[i] = RSConfig::NUM_L2_PER_L1;
      }
    }
    // transfer to device
    gpuErrchk(cudaMallocAsync(&d_num_last_l2_blocks_,
                              num_last_l2_blocks.size() * sizeof(uint16_t),
                              cudaStreamDefault));
    gpuErrchk(cudaMemcpyAsync(d_num_last_l2_blocks_, num_last_l2_blocks.data(),
                              num_last_l2_blocks.size() * sizeof(uint16_t),
                              cudaMemcpyHostToDevice, cudaStreamDefault));

    std::vector<size_t> num_l2_blocks_per_arr(num_arrays);
    for (size_t i = 0; i < num_arrays; ++i) {
      num_l2_blocks_per_arr[i] =
          (num_l1_blocks_[i] - 1) * RSConfig::NUM_L2_PER_L1 +
          num_last_l2_blocks[i];
    }

    total_num_l2_blocks_ = 0;
    for (auto const& num_blocks : num_l2_blocks_per_arr) {
      total_num_l2_blocks_ += num_blocks;
    }

    // Allocate memory for the L2 index.
    // For convenience, right now first entry is for the first block, which is
    // always 0.
    gpuErrchk(cudaMallocAsync(&d_l2_indices_,
                              total_num_l2_blocks_ * sizeof(RSConfig::L2_TYPE),
                              cudaStreamDefault));
    gpuErrchk(cudaMemsetAsync(d_l2_indices_, 0,
                              total_num_l2_blocks_ * sizeof(RSConfig::L2_TYPE),
                              cudaStreamDefault));

    std::exclusive_scan(num_l2_blocks_per_arr.begin(),
                        num_l2_blocks_per_arr.end(),
                        num_l2_blocks_per_arr.begin(), 0);

    gpuErrchk(cudaMallocAsync(&d_l2_offsets_,
                              num_l2_blocks_per_arr.size() * sizeof(size_t),
                              cudaStreamDefault));
    gpuErrchk(cudaMemcpyAsync(d_l2_offsets_, num_l2_blocks_per_arr.data(),
                              num_l2_blocks_per_arr.size() * sizeof(size_t),
                              cudaMemcpyHostToDevice, cudaStreamDefault));

    // OPT: loop unnecessary for wavelet tree, since array sizes are
    // monotonically decreasing
    // Get maximum storage needed for device sums
    size_t temp_storage_bytes = 0;
    for (uint32_t i = 0; i < num_arrays; i++) {
      auto const num_l1_blocks = num_l1_blocks_[i];
      RSConfig::L1_TYPE* d_data = getL1EntryPointer(i, 0);
      size_t prev_storage_bytes = temp_storage_bytes;
      gpuErrchk(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                              d_data, num_l1_blocks));
      temp_storage_bytes = std::max(temp_storage_bytes, prev_storage_bytes);
    }
    void* d_temp_storage = nullptr;
    gpuErrchk(cudaMallocAsync(&d_temp_storage, temp_storage_bytes,
                              cudaStreamDefault));

    // Choose maximum possible items per thread
    struct cudaFuncAttributes funcAttrib;
    gpuErrchk(cudaFuncGetAttributes(&funcAttrib, calculateL2EntriesKernel));

    auto max_block_size =
        std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

    max_block_size = findLargestDivisor(kMaxTPB, max_block_size);

    auto const& prop = getDeviceProperties();

    auto min_block_size = kMinTPB;
    while (prop.maxBlocksPerMultiProcessor * funcAttrib.sharedSizeBytes >
           prop.sharedMemPerMultiprocessor) {
      min_block_size += kMinTPB;
    }
    gpuErrchk(cudaMalloc(&d_total_num_ones_, num_arrays * sizeof(size_t)));
    kernelCheck();
#pragma omp parallel for num_threads(num_arrays)
    for (uint32_t i = 0; i < num_arrays; i++) {
      auto const num_l1_blocks = num_l1_blocks_[i];
      auto const num_l2_blocks =
          i == (num_arrays - 1)
              ? total_num_l2_blocks_ - num_l2_blocks_per_arr[i]
              : num_l2_blocks_per_arr[i + 1] - num_l2_blocks_per_arr[i];

      if (num_l1_blocks == 1) {
        uint32_t block_size =
            std::min(max_block_size, static_cast<uint32_t>(num_l2_blocks));

        // Round up to next WS
        if (block_size % WS != 0) {
          block_size += (WS - block_size % WS);
        }

        calculateL2EntriesKernel<<<1, block_size>>>(
            *this, i, num_l2_blocks, num_l1_blocks, block_size,
            (RSConfig::NUM_L2_PER_L1 + block_size - 1) / block_size,
            bit_array_.sizeHost(i));
        kernelStreamCheck(cudaStreamPerThread);
      } else {
        auto const& ideal_configs = getIdealConfigs(prop.name);
        uint32_t const block_size =
            ideal_configs.ideal_TPB_calculateL2EntriesKernel != 0
                ? ideal_configs.ideal_TPB_calculateL2EntriesKernel
                : min_block_size;
        // calculate L2 entries for all L1 blocks
        calculateL2EntriesKernel<<<num_l1_blocks, block_size>>>(
            *this, i, num_last_l2_blocks[i], num_l1_blocks, block_size,
            (RSConfig::NUM_L2_PER_L1 + block_size - 1) / block_size,
            bit_array_.sizeHost(i));
        kernelStreamCheck(cudaStreamPerThread);

        RSConfig::L1_TYPE* const d_data = getL1EntryPointer(i, 0);

#pragma omp critical
        {  // Run inclusive prefix sum
          gpuErrchk(cub::DeviceScan::InclusiveSum(
              d_temp_storage, temp_storage_bytes, d_data, num_l1_blocks));
          kernelCheck();
        }
      }
    }
    gpuErrchk(cudaFreeAsync(d_temp_storage, cudaStreamDefault));
    addNumOnesKernel<<<1, std::min(kMaxTPB, static_cast<uint32_t>(num_arrays)),
                       0, cudaStreamDefault>>>(*this, num_arrays);
    // Get the number of ones per bit array
    std::vector<size_t> num_ones_per_array(num_arrays);
    gpuErrchk(cudaMemcpyAsync(num_ones_per_array.data(), d_total_num_ones_,
                              num_arrays * sizeof(size_t),
                              cudaMemcpyDeviceToHost, cudaStreamDefault));
    std::vector<size_t> num_ones_samples_per_array(num_arrays);
    std::vector<size_t> num_zeros_samples_per_array(num_arrays);
    kernelCheck();
    for (uint8_t i = 0; i < num_arrays; i++) {
      num_ones_samples_per_array[i] =
          num_ones_per_array[i] / RSConfig::SELECT_SAMPLE_RATE;
      num_zeros_samples_per_array[i] =
          (bit_array_.sizeHost(i) - num_ones_per_array[i]) /
          RSConfig::SELECT_SAMPLE_RATE;
    }
    size_t const total_ones_samples =
        std::accumulate(num_ones_samples_per_array.begin(),
                        num_ones_samples_per_array.end(), 0);
    std::exclusive_scan(num_ones_samples_per_array.begin(),
                        num_ones_samples_per_array.end(),
                        num_ones_samples_per_array.begin(), 0);
    gpuErrchk(cudaMallocAsync(&d_select_samples_1_,
                              total_ones_samples * sizeof(size_t),
                              cudaStreamDefault));
    gpuErrchk(cudaMallocAsync(&d_select_samples_1_offsets_,
                              num_arrays * sizeof(size_t), cudaStreamDefault));
    gpuErrchk(cudaMemcpyAsync(d_select_samples_1_offsets_,
                              num_ones_samples_per_array.data(),
                              num_arrays * sizeof(size_t),
                              cudaMemcpyHostToDevice, cudaStreamDefault));

    size_t const total_zeros_samples =
        std::accumulate(num_zeros_samples_per_array.begin(),
                        num_zeros_samples_per_array.end(), 0);
    std::exclusive_scan(num_zeros_samples_per_array.begin(),
                        num_zeros_samples_per_array.end(),
                        num_zeros_samples_per_array.begin(), 0);
    gpuErrchk(cudaMallocAsync(&d_select_samples_0_,
                              total_zeros_samples * sizeof(size_t),
                              cudaStreamDefault));
    gpuErrchk(cudaMallocAsync(&d_select_samples_0_offsets_,
                              num_arrays * sizeof(size_t), cudaStreamDefault));
    gpuErrchk(cudaMemcpy(d_select_samples_0_offsets_,
                         num_zeros_samples_per_array.data(),
                         num_arrays * sizeof(size_t), cudaMemcpyHostToDevice));
    kernelCheck();

    if (total_ones_samples > 0 or total_zeros_samples > 0) {
      auto const& ideal_configs = getIdealConfigs(prop.name);
#pragma omp parallel for num_threads(num_arrays)
      for (uint8_t i = 0; i < num_arrays; i++) {
        size_t const num_ones_samples =
            i == num_arrays - 1
                ? total_ones_samples - num_ones_samples_per_array[i]
                : num_ones_samples_per_array[i + 1] -
                      num_ones_samples_per_array[i];
        size_t const num_zeros_samples =
            i == num_arrays - 1
                ? total_zeros_samples - num_zeros_samples_per_array[i]
                : num_zeros_samples_per_array[i + 1] -
                      num_zeros_samples_per_array[i];

        if (num_ones_samples > 0 or num_zeros_samples > 0) {
          size_t const num_warps =
              ideal_configs.ideal_tot_threads_calculateSelectSamplesKernel != 0
                  ? std::min(
                        num_ones_samples + num_zeros_samples,
                        ideal_configs
                                .ideal_tot_threads_calculateSelectSamplesKernel /
                            WS)
                  : std::min(
                        static_cast<size_t>((prop.maxThreadsPerMultiProcessor *
                                                 prop.multiProcessorCount +
                                             WS - 1) /
                                            WS),
                        num_ones_samples + num_zeros_samples);
          auto const [blocks, threads] =
              ideal_configs.ideal_TPB_calculateSelectSamplesKernel != 0
                  ? getLaunchConfig(
                        num_warps,
                        ideal_configs.ideal_TPB_calculateSelectSamplesKernel,
                        ideal_configs.ideal_TPB_calculateSelectSamplesKernel)
                  : getLaunchConfig(num_warps, kMinTPB, kMaxTPB);

          calculateSelectSamplesKernel<<<blocks, threads>>>(
              *this, i, blocks * threads, num_ones_samples, num_zeros_samples);
          kernelStreamCheck(cudaStreamPerThread);
        }
      }
    }
    kernelCheck();
  }

  /*!
   * \brief Copy constructor.
   * \param other \c RankSelect object to be copied.
   */
  __host__ RankSelect(RankSelect const& other) noexcept;

  /*!
   * \brief Move assignment operator.
   * \param other \c RankSelect object to be moved.
   */
  __host__ RankSelect& operator=(RankSelect&& other) noexcept;

  __host__ ~RankSelect();

  /*!
   * \brief Computes rank of zeros.
   * \tparam NumThreads Number of threads accessing the function.
   * \param array_index Index of the bit array to be used.
   * \param index Index the rank of zeros is computed for.
   * \return Number of zeros (rank) before position \c index, i.e. in the slice
   * [0, i).
   */
  template <int NumThreads, bool GetBit, int Value>
  __device__ [[nodiscard]] auto rank(uint8_t const array_index,
                                     size_t const index, size_t const offset) {
    static_assert(Value == 0 or Value == 1, "Value must be 0 or 1.");
    assert(array_index < bit_array_.numArrays());
    assert(index <= bit_array_.size(array_index));
    if (index == 0) {
      if constexpr (GetBit) {
        return RankResult{0, bit_array_.access(array_index, 0, offset)};
      } else {
        return size_t(0);
      }
    } else if (index == bit_array_.size(array_index)) {
      if constexpr (Value == 0) {
        if constexpr (GetBit) {
          return RankResult{index - d_total_num_ones_[array_index], false};
        } else {
          return index - d_total_num_ones_[array_index];
        }
      } else {
        if constexpr (GetBit) {
          return RankResult{d_total_num_ones_[array_index], false};
        } else {
          return d_total_num_ones_[array_index];
        }
      }
    }
    uint8_t t_id = 0;
    if constexpr (NumThreads > 1) {
      t_id = threadIdx.x % NumThreads;
    }
    size_t const l1_pos = index / RSConfig::L1_BIT_SIZE;
    RSConfig::L2_TYPE const l2_pos =
        (index % RSConfig::L1_BIT_SIZE) / RSConfig::L2_BIT_SIZE;
    // Only first thread in the local block stores the result
    size_t result = (d_l1_indices_[d_l1_offsets_[array_index] + l1_pos] +
                     d_l2_indices_[d_l2_offsets_[array_index] +
                                   l1_pos * RSConfig::NUM_L2_PER_L1 + l2_pos]) *
                    (t_id == 0);
    size_t const start_word =
        l1_pos * RSConfig::L1_WORD_SIZE + l2_pos * RSConfig::L2_WORD_SIZE;
    size_t const end_word = index / (sizeof(uint32_t) * 8);

    uint64_t word;
    uint8_t bit_at_index;
    bool has_last_word = false;
    uint8_t const bit_index = index % (sizeof(uint64_t) * 8);
    for (size_t i = start_word + 2 * t_id; i <= end_word; i += 2 * NumThreads) {
      word = bit_array_.twoWords(array_index, i, offset);
      if (end_word == 0 or i >= end_word - 1) {
        // Only consider bits up to the index.
        if constexpr (GetBit) {
          bit_at_index = static_cast<uint8_t>((word >> bit_index) & 1U);
          has_last_word = true;
        }
        word = bit_array_.partialTwoWords(word, bit_index);
      }
      result += __popcll(word);
    }

    if constexpr (NumThreads > 1) {
      uint32_t mask = ~0;
      if constexpr (NumThreads < WS) {
        mask = ((1 << NumThreads) - 1)
               << (NumThreads * ((threadIdx.x % WS) / NumThreads));
      }
      result = warpReduce<size_t, NumThreads>(mask, result);
      // communicate the result to all threads
      shareVar<size_t>(t_id == 0, result, mask);

      if constexpr (GetBit) {
        shareVar<uint8_t>(has_last_word, bit_at_index, mask);
      }
    }

    if constexpr (Value == 0) {
      result = index - result;
    }
    if constexpr (GetBit) {
      return RankResult{result, static_cast<bool>(bit_at_index)};
    } else {
      return result;
    }
  }

  /*!
   * \brief Get position of i-th zero or one. Starting from 1.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param array_index Index of the bit array to be used.
   * \param i i-th zero or one.
   * \param local_t_id Thread ID, has to start at 0.
   * \param num_threads Number of threads accessing the function. Right now 32
   * is assumed.
   * \return Position of the i-th zero/one. If i is larger than the number of
   * zeros/ones, the function returns the size of the bit array.
   */
  template <uint32_t Value, int NumThreads, bool UseSamples = true>
  __device__ [[nodiscard]] size_t select(uint32_t const array_index, size_t i,
                                         size_t const BA_offset) {
    static_assert(Value == 0 or Value == 1, "Value must be 0 or 1.");
    static_assert(NumThreads <= 32);
    assert(array_index < bit_array_.numArrays());
    assert(i > 0);
    if constexpr (Value == 0) {
      if (i > (bit_array_.size(array_index) - d_total_num_ones_[array_index])) {
        return bit_array_.size(array_index);
      }
    } else {
      if (i > d_total_num_ones_[array_index]) {
        return bit_array_.size(array_index);
      }
    }

    uint8_t local_t_id = 0;
    uint32_t mask;
    if constexpr (NumThreads > 1) {
      local_t_id = threadIdx.x % NumThreads;

      mask = ~0;
      if constexpr (NumThreads < WS) {
        mask = ((1 << NumThreads) - 1)
               << (NumThreads * ((threadIdx.x % WS) / NumThreads));
      }
    }

    size_t const prev_sample_index = i / RSConfig::SELECT_SAMPLE_RATE;
    size_t result =
        1;  // 1 so that "start_l1_block" is not 0 if no sample found
    size_t next_sample_pos = bit_array_.size(array_index);
    if constexpr (UseSamples) {
      if (prev_sample_index > 0) {
        if constexpr (Value == 0) {
          result =
              d_select_samples_0_[d_select_samples_0_offsets_[array_index] +
                                  prev_sample_index -
                                  1];  // -1 since pos of 0th val is not saved
          if ((prev_sample_index + 1) * RSConfig::SELECT_SAMPLE_RATE <=
              getTotalNumVals<0>(array_index)) {
            next_sample_pos =
                d_select_samples_0_[d_select_samples_0_offsets_[array_index] +
                                    prev_sample_index];
          }
        } else {
          result =
              d_select_samples_1_[d_select_samples_1_offsets_[array_index] +
                                  prev_sample_index - 1];
          if ((prev_sample_index + 1) * RSConfig::SELECT_SAMPLE_RATE <=
              getTotalNumVals<1>(array_index)) {
            next_sample_pos =
                d_select_samples_1_[d_select_samples_1_offsets_[array_index] +
                                    prev_sample_index];
          }
        }
      } else {
        if constexpr (Value == 0) {
          if (RSConfig::SELECT_SAMPLE_RATE > getTotalNumVals<0>(array_index)) {
            next_sample_pos = bit_array_.size(array_index);
          } else {
            next_sample_pos =
                d_select_samples_0_[d_select_samples_0_offsets_[array_index]];
          }
        } else {
          if (RSConfig::SELECT_SAMPLE_RATE > getTotalNumVals<1>(array_index)) {
            next_sample_pos = bit_array_.size(array_index);
          } else {
            next_sample_pos =
                d_select_samples_1_[d_select_samples_1_offsets_[array_index]];
          }
        }
      }
    }

    //  Find next L1 block where the i-th zero/one is, starting from result
    size_t const l1_offset = d_l1_offsets_[array_index];
    size_t const l2_offset = d_l2_offsets_[array_index];

    size_t const num_l1_blocks = d_num_l1_blocks_[array_index];
    size_t has_i_before = 0;

    if (num_l1_blocks > 1) {
      // Starting from 1 since 0-th entry always 0.
      size_t const start_l1_block =
          (result + RSConfig::L1_BIT_SIZE - 1) / RSConfig::L1_BIT_SIZE;
      size_t const end_l1_block =
          min(num_l1_blocks, ((next_sample_pos + RSConfig::L1_BIT_SIZE - 1) /
                              RSConfig::L1_BIT_SIZE) +
                                 1);
      if constexpr (not UseSamples) {
        assert(end_l1_block == num_l1_blocks);
        assert(start_l1_block == 1);
      }

      uint32_t const l1_blocks_per_thread =
          ((end_l1_block - start_l1_block) + NumThreads - 1) / NumThreads;

      size_t const local_start_l1_block =
          start_l1_block + local_t_id * l1_blocks_per_thread;
      size_t const local_end_l1_block =
          min(local_start_l1_block + l1_blocks_per_thread, end_l1_block);

      if (local_start_l1_block < local_end_l1_block) {
        if constexpr (Value == 0) {
          has_i_before =
              thrust::lower_bound(
                  thrust::seq, d_l1_indices_ + l1_offset + local_start_l1_block,
                  d_l1_indices_ + l1_offset + local_end_l1_block, i,
                  [&](auto const& elem, auto const& val) {
                    auto const num_bits =
                        reinterpret_cast<uintptr_t>(&elem - d_l1_indices_ -
                                                    l1_offset) *
                        RSConfig::L1_BIT_SIZE;
                    return (num_bits - elem) < val;
                  }) -
              (d_l1_indices_ + l1_offset);
        } else {
          has_i_before =
              thrust::lower_bound(
                  thrust::seq, d_l1_indices_ + l1_offset + local_start_l1_block,
                  d_l1_indices_ + l1_offset + local_end_l1_block, i) -
              (d_l1_indices_ + l1_offset);
        }
        if (has_i_before == local_end_l1_block) {
          has_i_before = 0;
        }
      }

      if constexpr (NumThreads > 1) {
        shareVar<size_t>(has_i_before != 0, has_i_before, mask);
      }
    }
    if (has_i_before != 0) {
      size_t const curr_l1_block = has_i_before - 1;
      result = curr_l1_block * RSConfig::L1_BIT_SIZE;
      if constexpr (Value == 0) {
        i -= curr_l1_block * RSConfig::L1_BIT_SIZE -
             d_l1_indices_[l1_offset + curr_l1_block];
      } else {
        i -= d_l1_indices_[l1_offset + curr_l1_block];
      }
    } else {
      result = (num_l1_blocks - 1) * RSConfig::L1_BIT_SIZE;
      if constexpr (Value == 0) {
        i -= (num_l1_blocks - 1) * RSConfig::L1_BIT_SIZE -
             d_l1_indices_[l1_offset + num_l1_blocks - 1];
      } else {
        i -= d_l1_indices_[l1_offset + num_l1_blocks - 1];
      }
    }

    bool const is_last_l1_block = has_i_before == 0;
    RSConfig::L2_TYPE const l1_block_length =
        is_last_l1_block ? d_num_last_l2_blocks_[array_index]
                         : RSConfig::NUM_L2_PER_L1;
    size_t const l1_block_start =
        (result / RSConfig::L1_BIT_SIZE) * RSConfig::NUM_L2_PER_L1;
    RSConfig::L2_TYPE const l2_blocks_per_thread =
        (l1_block_length + NumThreads - 1) / NumThreads;

    size_t const start_l2_block =
        l1_block_start + local_t_id * l2_blocks_per_thread;
    size_t const end_l2_block = min(l1_block_start + l1_block_length,
                                    start_l2_block + l2_blocks_per_thread);

    has_i_before = 0;
    if (start_l2_block < end_l2_block) {
      if constexpr (Value == 0) {
        has_i_before =
            thrust::lower_bound(
                thrust::seq, d_l2_indices_ + l2_offset + start_l2_block,
                d_l2_indices_ + l2_offset + end_l2_block, i,
                [&](auto const& elem, auto const& val) {
                  auto const num_bits =
                      reinterpret_cast<uintptr_t>(&elem - d_l2_indices_ -
                                                  l2_offset - l1_block_start) *
                      RSConfig::L2_BIT_SIZE;
                  return (num_bits - elem) < val;
                }) -
            (d_l2_indices_ + l2_offset);
      } else {
        has_i_before =
            thrust::lower_bound(thrust::seq,
                                d_l2_indices_ + l2_offset + start_l2_block,
                                d_l2_indices_ + l2_offset + end_l2_block, i) -
            (d_l2_indices_ + l2_offset);
      }
      if (has_i_before == end_l2_block) {
        has_i_before = 0;
      }
    }

    if constexpr (NumThreads > 1) {
      shareVar<size_t>(has_i_before != 0, has_i_before, mask);
    }
    if (has_i_before != 0) {
      result += (has_i_before - l1_block_start - 1) * RSConfig::L2_BIT_SIZE;
      if constexpr (Value == 0) {
        i -= (has_i_before - l1_block_start - 1) * RSConfig::L2_BIT_SIZE -
             d_l2_indices_[l2_offset + l1_block_start +
                           (has_i_before - l1_block_start - 1)];
      } else {
        i -= d_l2_indices_[l2_offset + l1_block_start +
                           (has_i_before - l1_block_start - 1)];
      }
    } else {
      result += (l1_block_length - 1) * RSConfig::L2_BIT_SIZE;
      if constexpr (Value == 0) {
        i -= (l1_block_length - 1) * RSConfig::L2_BIT_SIZE -
             d_l2_indices_[l2_offset + l1_block_start + l1_block_length - 1];
      } else {
        i -= d_l2_indices_[l2_offset + l1_block_start + l1_block_length - 1];
      }
    }

    RSConfig::L2_TYPE const l2_block_length =
        (has_i_before == 0 and is_last_l1_block)
            ? bit_array_.sizeInWords(array_index) -
                  (result / (sizeof(uint32_t) * 8))
            : RSConfig::L2_WORD_SIZE;

    size_t const word_start = result / (sizeof(uint32_t) * 8);
    result = 0;  // 0 signalizes that nothing was found
    if constexpr (NumThreads > 1) {
      for (size_t current_group_word = word_start;
           current_group_word < word_start + l2_block_length;
           current_group_word += 2 * NumThreads) {
        size_t const local_word = current_group_word + 2 * local_t_id;
        uint64_t word;
        if constexpr (Value == 0) {
          word = ~0ULL;
        } else {
          word = 0ULL;
        }
        RSConfig::L2_TYPE num_vals = 0;
        if (local_word < word_start + l2_block_length) {
          word = bit_array_.twoWords(array_index, local_word, BA_offset);
          if constexpr (Value == 0) {
            num_vals = (sizeof(uint64_t) * 8) - __popcll(word);
          } else {
            num_vals = __popcll(word);
          }
        }
        RSConfig::L2_TYPE cum_vals = 0;

        // inclusive prefix sum
        warpSum<RSConfig::L2_TYPE, NumThreads, true>(num_vals, cum_vals, mask);

        // Check if the i-th zero/one is in the current word
        RSConfig::L2_TYPE const vals_at_start = cum_vals - num_vals;
        if (cum_vals >= i and vals_at_start < i) {
          // Find the position of the i-th zero/one in the word
          // 1-indexed to distinguish from having found nothing, which is
          // 0.
          i -= vals_at_start;
          result = local_word * (sizeof(uint32_t) * 8) +
                   getNBitPos<Value, uint64_t>(i, word) + 1;
        }
        // communicate the result to all threads
        shareVar<size_t>(result != 0, result, mask);
        if (result != 0) {
          break;
        }
        // if no result found, update i
        i -= cum_vals;
        shareVar<size_t>(local_t_id == (NumThreads - 1), i, mask);
      }
    } else {
      RSConfig::L2_TYPE num_vals = 0;
      for (size_t j = 0; j < l2_block_length; j += 2) {
        uint64_t word;
        if constexpr (Value == 0) {
          word = ~0ULL;
        } else {
          word = 0ULL;
        }
        RSConfig::L2_TYPE current_num_vals = 0;

        word = bit_array_.twoWords(array_index, word_start + j, BA_offset);
        if constexpr (Value == 0) {
          current_num_vals = (sizeof(uint64_t) * 8) - __popcll(word);
        } else {
          current_num_vals = __popcll(word);
        }
        if ((num_vals + current_num_vals) >= i and num_vals < i) {
          i -= num_vals;
          result = (word_start + j) * (sizeof(uint32_t) * 8) +
                   getNBitPos<Value, uint64_t>(i, word) + 1;
          break;
        }
        num_vals += current_num_vals;
      }
    }

    // If nothing found, return the size of the array
    if (result == 0 or result > bit_array_.size(array_index)) {
      result = bit_array_.size(array_index);
    } else {
      result--;
    }
    return result;
  }

  /*!
   * \brief Get the number of L1 blocks for a bit array.
   * \param array_index Index of the bit array to be used.
   * \return Number of L1 blocks.
   */
  __device__ [[nodiscard]] size_t getNumL1Blocks(
      uint32_t const array_index) const;

  /*!
   * \brief Get the number of L2 blocks for a bit array.
   * \param array_index Index of the bit array to be used.
   * \return Number of L2 blocks.
   */
  __device__ [[nodiscard]] size_t getNumL2Blocks(
      uint32_t const array_index) const;

  /*!
   * \brief Get the number of L2 blocks in the last L1 block for a bit array.
   * \param array_index Index of the bit array to be used.
   * \return Number of L2 blocks in the last L1 block.
   */
  __device__ [[nodiscard]] size_t getNumLastL2Blocks(
      uint32_t const array_index) const;

  template <uint32_t Value>
  __device__ [[nodiscard]] size_t getTotalNumVals(
      uint32_t const array_index) const {
    static_assert(Value == 0 or Value == 1, "Value must be 0 or 1.");
    if constexpr (Value == 1) {
      return d_total_num_ones_[array_index];
    } else {
      return bit_array_.size(array_index) - d_total_num_ones_[array_index];
    }
  }

  template <uint32_t Value>
  __device__ [[nodiscard]] size_t getSelectSample(uint32_t const array_index,
                                                  size_t const index) const {
    if constexpr (Value == 0) {
      return d_select_samples_0_[d_select_samples_0_offsets_[array_index] +
                                 index];
    } else {
      return d_select_samples_1_[d_select_samples_1_offsets_[array_index] +
                                 index];
    }
  }

  /*!
   * \brief Write a value to the L2 index.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L2 index to be written to.
   * \param value Value to be written.
   */
  __device__ void writeL2Index(uint32_t const array_index, size_t const index,
                               RSConfig::L2_TYPE const value) noexcept;

  /*!
   * \brief Write a value to the L1 index.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L1 index to be written to.
   * \param value Value to be written.
   */
  __device__ void writeL1Index(uint32_t const array_index, size_t const index,
                               RSConfig::L1_TYPE const value) noexcept;

  __device__ void writeTotalNumOnes(uint32_t const array_index,
                                    size_t const value) noexcept {
    assert(array_index < bit_array_.numArrays());
    d_total_num_ones_[array_index] = value;
  }

  template <uint32_t Value>
  __device__ void writeSelectSample(uint32_t const array_index,
                                    size_t const index,
                                    size_t const value) noexcept {
    static_assert(Value == 0 or Value == 1, "Value must be 0 or 1.");
    if constexpr (Value == 0) {
      d_select_samples_0_[d_select_samples_0_offsets_[array_index] + index] =
          value;
    } else {
      d_select_samples_1_[d_select_samples_1_offsets_[array_index] + index] =
          value;
    }
  }

  /*!
   * \brief Get an L1 entry for a bit array.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L1 entry to be returned.
   * \return L1 entry.
   */
  __device__ [[nodiscard]] RSConfig::L1_TYPE getL1Entry(
      uint32_t const array_index, size_t const index) const;

  /*!
   * \brief Get a device pointer to an L1 entry for a bit array.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L1 entry to be returned.
   * \return Pointer to L1 entry.
   */
  __host__ [[nodiscard]] RSConfig::L1_TYPE* getL1EntryPointer(
      uint32_t const array_index, size_t const index) const;

  /*!
   * \brief Get an L2 entry for a bit array.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L2 entry to be returned.
   * \return L2 entry.
   */
  __device__ [[nodiscard]] size_t getL2Entry(uint32_t const array_index,
                                             size_t const index) const;

  /*!
   * \brief Write the number of L2 blocks in the last L1 block of a bit array.
   * \param array_index Index of the bit array to be used.
   * \param value Number of L2 blocks in the last L1 block.
   */
  __device__ void writeNumLastL2Blocks(uint32_t const array_index,
                                       uint16_t const value) noexcept;

  /*!
   * \brief Do a sum reduction with a subset of a warp.
   * \tparam T Type of the value to be summed.
   * \tparam NumThreads Number of threads accessing the function.
   * \param mask Mask representing the threads that should participate in the
   * reduction.
   * \param val Value to be summed.
   * \return Sum of the values of the threads in the mask. Only valid for the
   * first thread in the mask.
   */

  template <typename T, int NumThreads>
  __device__ T warpReduce(uint32_t const mask, T val) {
#pragma unroll
    for (int offset = NumThreads / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
    }
    return val;
  }

  template <typename T, int NumThreads, bool IsInclusive>
  __device__ void warpSum(T& input, T& output, uint32_t const mask) {
    T val = input;
    uint8_t const lane = threadIdx.x % NumThreads;

#pragma unroll
    for (int offset = 1; offset < NumThreads; offset *= 2) {
      T temp = __shfl_up_sync(mask, val, offset);
      if (lane >= offset) {
        val += temp;
      }
    }

    if constexpr (IsInclusive) {
      output = val;
    } else {
      output = val - input;
    }
  }

  /*!
   * \brief Get the position of the n-th 0 or 1 bit in a word, starting from
   * the least significant bit.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param n Rank of the bit. Starts from 1.
   * \param word Word the bit is in.
   * \return Position of the n-th bit. Starts from 0.
   */
  template <uint32_t Value, typename T>
  __device__ [[nodiscard]] uint8_t getNBitPos(uint8_t n, T word) {
    static_assert(Value == 0 or Value == 1,
                  "Template parameter must be 0 or 1");
    static_assert(std::is_integral<T>::value, "T must be an integral type.");
    static_assert(sizeof(T) == 4 or sizeof(T) == 8, "T must be 4 or 8 bytes.");
    assert(n > 0);
    if constexpr (Value == 0) {
      word = ~word;
    }
    uint32_t mask = 0x0000FFFFu;
    uint8_t shift = 16u;
    uint8_t base = 0u;
    uint8_t count;
    if constexpr (sizeof(T) == 4) {
      assert(n <= 32);
    } else {
      assert(n <= 64);
      count = __popc(word & 0xFFFFFFFFu);
      if (n > count) {
        base = 32;
        n -= count;
        // move top 32 bits to bottom
        word >>= 32;
      }
    }

#pragma unroll
    for (uint8_t i = 0; i < 5; i++) {
      count = __popc(word & mask);
      if (n > count) {
        base += shift;
        shift >>= 1;
        mask |= mask << shift;
      } else {
        shift >>= 1;
        mask >>= shift;
      }
    }
    return base;
  }

  __host__ [[nodiscard]] static size_t getNeededGPUMemory(
      size_t const size, uint8_t const num_arrays) noexcept;

 private:
  // TODO: num_l1_blocks and l2 blocks not necessary
  RSConfig::L1_TYPE* d_l1_indices_ =
      nullptr; /*!< Device pointer to L1 indices for all arrays.*/
  RSConfig::L2_TYPE* d_l2_indices_ =
      nullptr; /*!< Device pointer to L2 indices for all arrays.*/

  size_t* d_l1_offsets_ = nullptr; /*!< Offsets where each L1 index for a bit
                            array starts.*/
  size_t* d_l2_offsets_ = nullptr; /*!< Offsets where each L2 index for a bit
                            array starts.*/

  uint16_t* d_num_last_l2_blocks_ = nullptr; /*!< Number of L2 blocks in the
                                   last L1 block for each bit array.*/
  size_t* d_num_l1_blocks_ =
      nullptr; /*!< Number of L1 blocks for each bit array.*/
  std::vector<size_t> num_l1_blocks_; /*!< Number of L1 blocks for all bit
                                      arrays. Not accessible from device.*/
  size_t total_num_l2_blocks_;        /*!< Total number of L2 blocks for all bit
                                         arrays.*/

  size_t* d_select_samples_0_ =
      nullptr; /*!< Sampled positions for select queries.*/
  size_t* d_select_samples_0_offsets_ = nullptr; /*!< Offsets where each array's
                                    select samples start.*/
  size_t* d_select_samples_1_ =
      nullptr; /*!< Sampled positions for select queries.*/
  size_t* d_select_samples_1_offsets_ = nullptr; /*!< Offsets where each array's
                                    select samples start.*/
  size_t* d_total_num_ones_ =
      nullptr; /*!< Total number of ones for each array.*/

  bool is_copy_; /*!< Flag to signal whether current object is a
                    copy.*/
};  // class RankSelect

/*!
 * \brief Fill L2 indices and prepare L1 indices for prefix sum.
 * \param rank_select RankSelect object to fill indices for.
 * \param array_index Index of the bit array to be used.
 * \param num_last_l2_blocks Number of L2 blocks in the last L1 block.
 */
__global__ LB(MAX_TPB, MIN_BPM) static void calculateL2EntriesKernel(
    RankSelect rank_select, uint32_t const array_index,
    uint16_t const num_last_l2_blocks, size_t const num_l1_blocks,
    uint32_t const num_threads, uint32_t const num_iters,
    size_t const bit_array_size) {
  assert(blockDim.x % WS == 0);
  static_assert(RSConfig::NUM_L2_PER_L1 % WS == 0);
  __shared__ RSConfig::L2_TYPE l2_entries[RSConfig::NUM_L2_PER_L1];

  auto t_id = threadIdx.x;
  if (blockIdx.x < gridDim.x - 1) {
    size_t const offset = rank_select.bit_array_.getOffset(array_index);
    // find L1 block index
    auto const l1_index = blockIdx.x;

    for (uint32_t i = t_id; i < RSConfig::NUM_L2_PER_L1; i += num_threads) {
      RSConfig::L2_TYPE num_ones = 0;
      size_t const start_word =
          l1_index * RSConfig::L1_WORD_SIZE + i * RSConfig::L2_WORD_SIZE;

      size_t const end_word = start_word + RSConfig::L2_WORD_SIZE;
      for (size_t j = start_word; j < end_word; j += 2) {
        // Global memory load
        // load as 64 bits.
        uint64_t const word =
            rank_select.bit_array_.twoWords(array_index, j, offset);
        num_ones += __popcll(word);
      }
      l2_entries[i] = num_ones;
    }

    __syncthreads();
    auto warp_id = threadIdx.x / WS;
    RSConfig::L2_TYPE l2_entry_sum, l2_entry;
    for (auto i = 0; i < num_iters; i++) {
      if (warp_id < RSConfig::NUM_L2_PER_L1 / WS) {
        l2_entry = l2_entries[t_id];
        rank_select.warpSum<RSConfig::L2_TYPE, WS, false>(l2_entry,
                                                          l2_entry_sum, ~0);
        __syncwarp();
      }
      __syncthreads();
      if (warp_id < RSConfig::NUM_L2_PER_L1 / WS) {
        // Last thread in warp writes aggregated value to shmem
        if (t_id % WS == WS - 1) {
          l2_entries[warp_id] = l2_entry_sum + l2_entry;
        }
      }
      __syncthreads();
      if (warp_id < RSConfig::NUM_L2_PER_L1 / WS) {
        // Get aggregates from previous warps and sum them to own result
        for (auto j = 0; j < warp_id; j++) {
          l2_entry_sum += l2_entries[j];
        }
        // Write result to global memory
        rank_select.writeL2Index(array_index,
                                 l1_index * RSConfig::NUM_L2_PER_L1 + t_id,
                                 l2_entry_sum);
        if (t_id == RSConfig::NUM_L2_PER_L1 - 1) {
          rank_select.writeL1Index(array_index, l1_index + 1,
                                   l2_entry_sum + l2_entry);
        }
      }
      warp_id += blockDim.x / WS;
      t_id += blockDim.x;
    }
  }

  else {
    if (threadIdx.x == 0) {
      rank_select.writeNumLastL2Blocks(array_index, num_last_l2_blocks);
    }

    auto const l1_start_word =
        (rank_select.getNumL1Blocks(array_index) - 1) * RSConfig::L1_WORD_SIZE;

    for (uint32_t i = t_id; i < num_last_l2_blocks; i += num_threads) {
      RSConfig::L2_TYPE num_ones = 0;
      size_t const start_word = l1_start_word + i * RSConfig::L2_WORD_SIZE;

      size_t const end_word =
          min(rank_select.bit_array_.sizeInWords(array_index),
              start_word + RSConfig::L2_WORD_SIZE);
      for (size_t j = start_word; j < end_word; j++) {
        uint32_t word = rank_select.bit_array_.word(array_index, j);

        if (j == (bit_array_size + sizeof(uint32_t) * 8 - 1) /
                         (sizeof(uint32_t) * 8) -
                     1) {
          uint8_t const bit_index = bit_array_size % (sizeof(uint32_t) * 8);
          if (bit_index != 0) {
            word = rank_select.bit_array_.partialWord(word, bit_index);
          }
        }
        num_ones += __popc(word);
      }
      l2_entries[i] = num_ones;
    }

    __syncthreads();
    auto warp_id = threadIdx.x / WS;
    RSConfig::L2_TYPE l2_entry_sum, l2_entry;
    uint32_t const needed_iters =
        (num_last_l2_blocks + blockDim.x - 1) / blockDim.x;
    uint8_t needed_warps = (num_last_l2_blocks + WS - 1) / WS;
    for (auto i = 0; i < needed_iters; i++) {
      if (warp_id < needed_warps) {
        l2_entry = l2_entries[t_id];
        rank_select.warpSum<RSConfig::L2_TYPE, WS, false>(l2_entry,
                                                          l2_entry_sum, ~0);
        __syncwarp();
      }
      __syncthreads();
      if (warp_id < needed_warps) {
        // Last thread in warp writes aggregated value to shmem
        if (t_id < num_last_l2_blocks and
            (t_id % WS == WS - 1 or t_id == num_last_l2_blocks - 1)) {
          l2_entries[warp_id] = l2_entry_sum + l2_entry;
        }
      }
      __syncthreads();
      if (warp_id < needed_warps) {
        // Get aggregates from previous warps and sum them to own result
        for (auto j = 0; j < warp_id; j++) {
          l2_entry_sum += l2_entries[j];
        }
        // Write result to global memory
        if (t_id < num_last_l2_blocks) {
          rank_select.writeL2Index(
              array_index, (num_l1_blocks - 1) * RSConfig::NUM_L2_PER_L1 + t_id,
              l2_entry_sum);
        }
        if (t_id == num_last_l2_blocks - 1) {
          rank_select.writeTotalNumOnes(array_index, l2_entry_sum + l2_entry);
        }
      }
      warp_id += blockDim.x / WS;
      t_id += blockDim.x;
    }
  }
  return;
}

__global__ LB(MAX_TPB, MIN_BPM) static void calculateSelectSamplesKernel(
    RankSelect rank_select, uint32_t const array_index,
    size_t const num_threads, size_t const num_ones_samples,
    size_t const num_zeros_samples) {
  assert(blockDim.x % WS == 0);

  size_t const global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t const offset = rank_select.bit_array_.getOffset(array_index);
  // 0-th sample not taken
  for (size_t i = global_t_id; i < num_ones_samples; i += num_threads) {
    size_t const pos = rank_select.select<1, 1, false>(
        array_index, (i + 1) * RSConfig::SELECT_SAMPLE_RATE, offset);

    assert(pos < rank_select.bit_array_.size(array_index));
    rank_select.writeSelectSample<1>(array_index, i, pos);
  }

  for (size_t i = global_t_id; i < num_zeros_samples; i += num_threads) {
    size_t const pos = rank_select.select<0, 1, false>(
        array_index, (i + 1) * RSConfig::SELECT_SAMPLE_RATE, offset);

    assert(pos < rank_select.bit_array_.size(array_index));
    rank_select.writeSelectSample<0>(array_index, i, pos);
  }
}

__global__ LB(MAX_TPB,
              MIN_BPM) static void addNumOnesKernel(RankSelect rank_select,
                                                    uint32_t const num_arrays) {
  size_t const global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_t_id < num_arrays) {
    auto const total_num_ones =
        rank_select.getTotalNumVals<1>(global_t_id) +
        rank_select.getL1Entry(global_t_id,
                               rank_select.getNumL1Blocks(global_t_id) - 1);
    rank_select.writeTotalNumOnes(global_t_id, total_num_ones);
  }
}
}  // namespace ecl
