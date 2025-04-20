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
#include <omp.h>

#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <mutex>
#include <numeric>
#include <queue>
#include <vector>

#include "ecl_wavelet/bitarray/bit_array.cuh"
#include "ecl_wavelet/bitarray/rank_select.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {

using namespace detail;

__host__ RankSelect::RankSelect(BitArray&& bit_array,
                                uint32_t const GPU_index) noexcept
    : bit_array_(std::move(bit_array)), is_copy_(false) {
  utils::checkWarpSize(GPU_index);

  auto const num_arrays = bit_array_.numArrays();

  // Compute the number of L1 blocks.
  num_l1_blocks_.resize(num_arrays);
  for (size_t i = 0; i < num_arrays; ++i) {
    num_l1_blocks_[i] = (bit_array_.size(i) + RSConfig::L1_BIT_SIZE - 1) /
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
                      l1_offsets.begin(), 0ULL);
  gpuErrchk(cudaMallocAsync(&d_l1_offsets_, l1_offsets.size() * sizeof(size_t),
                            cudaStreamDefault));
  gpuErrchk(cudaMemcpyAsync(d_l1_offsets_, l1_offsets.data(),
                            l1_offsets.size() * sizeof(size_t),
                            cudaMemcpyHostToDevice, cudaStreamDefault));

  // Get how many l2 blocks each last L1 block has
  std::vector<uint16_t> num_last_l2_blocks(num_arrays);
  for (size_t i = 0; i < num_arrays; ++i) {
    num_last_l2_blocks[i] = (bit_array_.size(i) % RSConfig::L1_BIT_SIZE +
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
                      num_l2_blocks_per_arr.begin(), 0ULL);

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
    gpuErrchk(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_data,
                                            num_l1_blocks));
    temp_storage_bytes = std::max(temp_storage_bytes, prev_storage_bytes);
  }
  void* d_temp_storage = nullptr;
  gpuErrchk(
      cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cudaStreamDefault));

  // Choose maximum possible items per thread
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, calculateL2EntriesKernel));

  auto max_block_size = std::min(
      utils::kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  max_block_size = utils::findLargestDivisor(utils::kMaxTPB, max_block_size);

  auto const& prop = utils::getDeviceProperties();

  auto min_block_size = utils::kMinTPB;
  while (prop.maxBlocksPerMultiProcessor * funcAttrib.sharedSizeBytes >
         prop.sharedMemPerMultiprocessor) {
    min_block_size += utils::kMinTPB;
  }
  gpuErrchk(cudaMalloc(&d_total_num_ones_, num_arrays * sizeof(size_t)));
  kernelCheck();
#pragma omp parallel for num_threads(num_arrays)
  for (uint32_t i = 0; i < num_arrays; i++) {
    gpuErrchk(cudaSetDevice(GPU_index));
    auto const num_l1_blocks = num_l1_blocks_[i];
    auto const num_l2_blocks =
        i == (num_arrays - 1u)
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
          *this, i, num_l2_blocks, num_l1_blocks,
          (RSConfig::NUM_L2_PER_L1 + block_size - 1) / block_size,
          bit_array_.size(i));
      kernelStreamCheck(cudaStreamPerThread);
    } else {
      uint32_t const block_size = min_block_size;
      // calculate L2 entries for all L1 blocks
      calculateL2EntriesKernel<<<num_l1_blocks, block_size>>>(
          *this, i, num_last_l2_blocks[i], num_l1_blocks,
          (RSConfig::NUM_L2_PER_L1 + block_size - 1) / block_size,
          bit_array_.size(i));
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
  addNumOnesKernel<<<
      1, std::min(utils::kMaxTPB, static_cast<uint32_t>(num_arrays)), 0,
      cudaStreamDefault>>>(*this, num_arrays);
  // Get the number of ones per bit array
  std::vector<size_t> num_ones_per_array(num_arrays);
  gpuErrchk(cudaMemcpyAsync(num_ones_per_array.data(), d_total_num_ones_,
                            num_arrays * sizeof(size_t), cudaMemcpyDeviceToHost,
                            cudaStreamDefault));
  std::vector<size_t> num_ones_samples_per_array(num_arrays);
  std::vector<size_t> num_zeros_samples_per_array(num_arrays);
  kernelCheck();
  for (uint8_t i = 0; i < num_arrays; i++) {
    num_ones_samples_per_array[i] =
        num_ones_per_array[i] / RSConfig::SELECT_SAMPLE_RATE;
    num_zeros_samples_per_array[i] =
        (bit_array_.size(i) - num_ones_per_array[i]) /
        RSConfig::SELECT_SAMPLE_RATE;
  }
  size_t const total_ones_samples =
      std::accumulate(num_ones_samples_per_array.begin(),
                      num_ones_samples_per_array.end(), 0ULL);
  std::exclusive_scan(num_ones_samples_per_array.begin(),
                      num_ones_samples_per_array.end(),
                      num_ones_samples_per_array.begin(), 0ULL);
  gpuErrchk(cudaMallocAsync(&d_select_samples_1_,
                            total_ones_samples * sizeof(size_t),
                            cudaStreamDefault));
  gpuErrchk(cudaMallocAsync(&d_select_samples_1_offsets_,
                            num_arrays * sizeof(size_t), cudaStreamDefault));
  gpuErrchk(cudaMemcpyAsync(
      d_select_samples_1_offsets_, num_ones_samples_per_array.data(),
      num_arrays * sizeof(size_t), cudaMemcpyHostToDevice, cudaStreamDefault));

  size_t const total_zeros_samples =
      std::accumulate(num_zeros_samples_per_array.begin(),
                      num_zeros_samples_per_array.end(), 0ULL);
  std::exclusive_scan(num_zeros_samples_per_array.begin(),
                      num_zeros_samples_per_array.end(),
                      num_zeros_samples_per_array.begin(), 0ULL);
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
#pragma omp parallel for num_threads(num_arrays)
    for (uint8_t i = 0; i < num_arrays; i++) {
      gpuErrchk(cudaSetDevice(GPU_index));

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
            std::min(static_cast<size_t>((prop.maxThreadsPerMultiProcessor *
                                              prop.multiProcessorCount +
                                          WS - 1) /
                                         WS),
                     num_ones_samples + num_zeros_samples);
        auto const [blocks, threads] =
            utils::getLaunchConfig(num_warps, utils::kMinTPB, utils::kMaxTPB);

        calculateSelectSamplesKernel<<<blocks, threads>>>(
            *this, i, blocks * threads, num_ones_samples, num_zeros_samples);
        kernelStreamCheck(cudaStreamPerThread);
      }
    }
  }
  kernelCheck();
}

__host__ RankSelect::RankSelect(RankSelect const& other) noexcept
    : bit_array_(other.bit_array_),
      d_l1_indices_(other.d_l1_indices_),
      d_l2_indices_(other.d_l2_indices_),
      d_l1_offsets_(other.d_l1_offsets_),
      d_l2_offsets_(other.d_l2_offsets_),
      d_num_last_l2_blocks_(other.d_num_last_l2_blocks_),
      d_num_l1_blocks_(other.d_num_l1_blocks_),
      num_l1_blocks_(other.num_l1_blocks_),
      total_num_l2_blocks_(other.total_num_l2_blocks_),
      d_select_samples_0_(other.d_select_samples_0_),
      d_select_samples_1_(other.d_select_samples_1_),
      d_select_samples_0_offsets_(other.d_select_samples_0_offsets_),
      d_select_samples_1_offsets_(other.d_select_samples_1_offsets_),
      d_total_num_ones_(other.d_total_num_ones_),
      is_copy_(true) {}

__host__ RankSelect& RankSelect::operator=(RankSelect&& other) noexcept {
  bit_array_ = std::move(other.bit_array_);
  d_l1_indices_ = other.d_l1_indices_;
  other.d_l1_indices_ = nullptr;
  d_l2_indices_ = other.d_l2_indices_;
  other.d_l2_indices_ = nullptr;
  d_l1_offsets_ = other.d_l1_offsets_;
  other.d_l1_offsets_ = nullptr;
  d_l2_offsets_ = other.d_l2_offsets_;
  other.d_l2_offsets_ = nullptr;
  d_num_last_l2_blocks_ = other.d_num_last_l2_blocks_;
  other.d_num_last_l2_blocks_ = nullptr;
  d_num_l1_blocks_ = other.d_num_l1_blocks_;
  other.d_num_l1_blocks_ = nullptr;
  num_l1_blocks_ = std::move(other.num_l1_blocks_);
  total_num_l2_blocks_ = other.total_num_l2_blocks_;
  d_select_samples_0_ = other.d_select_samples_0_;
  other.d_select_samples_0_ = nullptr;
  d_select_samples_1_ = other.d_select_samples_1_;
  other.d_select_samples_1_ = nullptr;
  d_select_samples_0_offsets_ = other.d_select_samples_0_offsets_;
  other.d_select_samples_0_offsets_ = nullptr;
  d_select_samples_1_offsets_ = other.d_select_samples_1_offsets_;
  other.d_select_samples_1_offsets_ = nullptr;
  d_total_num_ones_ = other.d_total_num_ones_;
  other.d_total_num_ones_ = nullptr;
  is_copy_ = other.is_copy_;
  other.is_copy_ = true;
  return *this;
}

__host__ RankSelect::~RankSelect() {
  if (not is_copy_) {
    gpuErrchk(cudaFree(d_l1_indices_));
    gpuErrchk(cudaFree(d_l2_indices_));
    gpuErrchk(cudaFree(d_l1_offsets_));
    gpuErrchk(cudaFree(d_l2_offsets_));
    gpuErrchk(cudaFree(d_num_last_l2_blocks_));
    gpuErrchk(cudaFree(d_num_l1_blocks_));
    gpuErrchk(cudaFree(d_select_samples_0_));
    gpuErrchk(cudaFree(d_select_samples_1_));
    gpuErrchk(cudaFree(d_select_samples_0_offsets_));
    gpuErrchk(cudaFree(d_select_samples_1_offsets_));
    gpuErrchk(cudaFree(d_total_num_ones_));
  }
}

__host__ [[nodiscard]] RSConfig::L1_TYPE* RankSelect::getL1EntryPointer(
    uint32_t const array_index, size_t const index) const noexcept {
  assert(array_index < bit_array_.numArrays());
  assert(index < num_l1_blocks_[array_index]);
  // Pointer arithmetic
  size_t offset = 0;
  for (size_t i = 0; i < array_index; ++i) {
    offset += num_l1_blocks_[i];
  }
  return d_l1_indices_ + offset + index;
}

__host__ [[nodiscard]] size_t RankSelect::getNeededGPUMemory(
    size_t const size, uint8_t const num_arrays) noexcept {
  size_t total_size = 0;
  total_size += num_arrays * sizeof(size_t);  // d_num_l1_blocks_
  size_t const num_l1_blocks_per_arr =
      (size + RSConfig::L1_BIT_SIZE - 1) / RSConfig::L1_BIT_SIZE;
  total_size +=
      num_l1_blocks_per_arr * num_arrays *
      sizeof(RSConfig::L1_TYPE);  // Upper bound of memory for d_l1_indices_
  total_size += num_arrays * sizeof(size_t);    // d_l1_offsets_
  total_size += num_arrays * sizeof(uint16_t);  // d_num_last_l2_blocks_
  total_size +=
      ((size + RSConfig::L2_BIT_SIZE - 1) / RSConfig::L2_BIT_SIZE) *
      num_arrays *
      sizeof(RSConfig::L2_TYPE);  // Upper bound of memory for d_l2_indices_
  total_size += num_arrays * sizeof(size_t);  // d_l2_offsets_
  size_t temp_storage_bytes = 0;
  RSConfig::L1_TYPE* d_data = nullptr;
  gpuErrchk(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_data,
                                          num_l1_blocks_per_arr));
  total_size += temp_storage_bytes;
  total_size += num_arrays * sizeof(size_t);  // d_total_num_ones_
  total_size += num_arrays * (size / RSConfig::SELECT_SAMPLE_RATE) *
                sizeof(size_t);  // d_select_samples_0_ and d_select_samples_1_
  total_size += num_arrays * sizeof(size_t);  // d_select_samples_0_offsets_
  total_size += num_arrays * sizeof(size_t);  // d_select_samples_1_offsets_
  return total_size;
}

namespace detail {
__global__ LB(MAX_TPB, MIN_BPM) void calculateL2EntriesKernel(
    RankSelect rank_select, uint32_t const array_index,
    uint16_t const num_last_l2_blocks, size_t const num_l1_blocks,
    uint32_t const num_iters, size_t const bit_array_size) {
  assert(blockDim.x % WS == 0);
  static_assert(RSConfig::NUM_L2_PER_L1 % WS == 0);
  __shared__ RSConfig::L2_TYPE l2_entries[RSConfig::NUM_L2_PER_L1];

  auto t_id = threadIdx.x;
  if (blockIdx.x < gridDim.x - 1) {
    size_t const offset = rank_select.bit_array_.getOffset(array_index);
    // find L1 block index
    auto const l1_index = blockIdx.x;

    for (uint32_t i = t_id; i < RSConfig::NUM_L2_PER_L1; i += blockDim.x) {
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

    for (uint32_t i = t_id; i < num_last_l2_blocks; i += blockDim.x) {
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

__global__ LB(MAX_TPB, MIN_BPM) void calculateSelectSamplesKernel(
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
              MIN_BPM) void addNumOnesKernel(RankSelect rank_select,
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
}  // namespace detail
}  // namespace ecl
