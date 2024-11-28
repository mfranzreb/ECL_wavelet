#pragma once

#include <algorithm>
#include <cmath>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <numeric>
#include <vector>

#include "bitarray/include/rank_select.cuh"

namespace ecl {

/*!
 * \brief Struct for creating a rank or select query.
 */
template <typename T>
struct RankSelectQuery {
  size_t index_;
  T* symbol_;
};

/*!
 * \brief Wavelet tree class.
 * \tparam T Type of the text data the tree will be built upon.
 */
template <typename T>
class WaveletTree {
 public:
  /*!
   * \brief Constructor. Builds the wavelet tree from the input data.
   * \param data Input data to build the wavelet tree.
   * \param alphabet Alphabet of the input data. Must be sorted.
   */
  // TODO: change to pointer so that it accepts strings
  __host__ WaveletTree(std::vector<T> data, std::vector<T>&& alphabet)
      : alphabet_(alphabet) {
    checkWarpSize();
    // make minimal alphabet
    alphabet_size_ = alphabet.size();
    std::vector<T> min_alphabet(alphabet_size_);
    std::iota(min_alphabet.begin(), min_alphabet.end(), 0);

    num_levels_ = ceil(log2(alphabet_size_));
    alphabet_start_bit_ = sizeof(T) * 8 - num_levels;

    // copy minimal alphabet to device
    gpuErrchk(cudaMalloc(&d_min_alphabet_, alphabet_size_ * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_min_alphabet_, min_alphabet.data(),
                         alphabet_size_ * sizeof(T), cudaMemcpyHostToDevice));

    // create codes and copy to device
    std::vector<Code> codes = createMinimalCodes(min_alphabet);
    gpuErrchk(cudaMalloc(&d_codes_, alphabet_size_ * sizeof(Code)));
    gpuErrchk(cudaMemcpy(d_codes_, codes.data(), alphabet_size_ * sizeof(Code),
                         cudaMemcpyHostToDevice));

    size_t data_size = data.size();
    // Copy data to device
    T* d_data;
    gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_data, data.data(), data_size * sizeof(T),
                         cudaMemcpyHostToDevice));

    // Allocate space for counts array
    gpuErrchk(cudaMalloc(&d_counts_, alphabet_size_ * sizeof(size_t)));
    gpuErrchk(cudaMemset(d_counts_, 0, alphabet_size_ * sizeof(size_t)));

    // Allocate space for alphabet
    T* d_alphabet_;
    gpuErrchk(cudaMalloc(&d_alphabet_, alphabet_size_ * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_alphabet_, alphabet.data(),
                         alphabet_size_ * sizeof(T), cudaMemcpyHostToDevice));

    // Compute global_histogram and change text to min_alphabet
    // TODO: find appropriate number of warps
    auto num_warps = std::min(data_size, 1'000'000);
    auto [num_blocks, threads_per_block] =
        getLaunchConfig(num_warps, 256, 1024);
    computeGlobalHistogramKernel<T><<<num_blocks, threads_per_block>>>(
        d_data, data_size, d_counts_, d_min_alphabet_, alphabet_size_);

    // Copy counts to host
    std::vector<size_t> counts(alphabet_size_);
    gpuErrchk(cudaMemcpy(counts.data(), d_counts_,
                         alphabet_size_ * sizeof(size_t),
                         cudaMemcpyDeviceToHost));

    // Calculate size of bit array at each level
    std::vector<size_t> bit_array_sizes(num_levels_, data_size);
    // Get min code length
    uint8_t min_code_len = codes[0].len_;
    for (auto const& code : codes) {
      std::min(min_code_len, code.len_);
    }
    for (size_t i = num_levels_ - 1; i >= min_code_len; --i) {
      for (size_t j = alphabet_size_ - 1; j >= 0; --j) {
        if (i >= codes[j].len_) {
          bit_array_sizes[i] -= counts[j];
        } else {
          break;
        }
      }
    }

    BitArray bit_array(bit_array_sizes, false);

    size_t local_hist_entries = 0;
    for (size_t i = 1; i < num_levels_; ++i) {
      local_hist_entries += powTwo<size_t>(i);
    }

    // TODO: maybe use 32 or 64 bits depending on text size
    // Allocate space for local histograms and borders
    size_t* d_local_histograms;
    gpuErrchk(cudaMalloc(&d_local_histograms,
                         local_hist_entries * num_warps * sizeof(size_t)));

    size_t* d_borders;
    gpuErrchk(cudaMalloc(&d_borders,
                         local_hist_entries * num_warps * sizeof(size_t)));

    // calculate local histograms
    buildLocalHistogramKernel<T><<<num_blocks, threads_per_block>>>(
        this, d_data, data_size, d_local_histograms, local_hist_entries,
        alphabet_start_bit_, bit_array);
    kernelCheck();

    // reduce all local histograms to global histogram in first warps borders
    std::vector<size_t> offsets(local_hist_entries + 1, num_warps);
    std::exclusive_scan(offsets.begin(), offsets.end() - 1, offsets.begin(), 0);
    offsets.back() = num_warps * local_hist_entries;
    size_t* d_offsets_it;
    gpuErrchk(cudaMalloc(&d_offsets_it, local_hist_entries * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(d_offsets_it, offsets.data(),
                         local_hist_entries * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(
        d_temp_storage, temp_storage_bytes, d_local_histograms, d_borders,
        local_hist_entries, d_offsets_it, d_offsets_it + 1);

    gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceSegmentedReduce::Sum(
        d_temp_storage, temp_storage_bytes, d_local_histograms, d_borders,
        local_hist_entries, d_offsets_it, d_offsets_it + 1);

    gpuErrchk(cudaFree(d_offsets_it));
    // Do exclusive prefix sum of first warp borders
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_borders,
                                  local_hist_entries);

    gpuErrchk(cudaFree(d_temp_storage));

    // communicate borders to other warps
    std::tie(num_blocks, threads_per_block) =
        getLaunchConfig(local_hist_entries / 32, 256, 1024);
    communicateBordersKernel<<<num_blocks, threads_per_block>>>(
        d_borders, d_local_histograms, local_hist_entries, num_warps);

    gpuErrchk(cudaFree(d_local_histograms));

    // allocate memory for sorted data
    T* sorted_data;
    gpuErrchk(cudaMalloc(&sorted_data, data_size * sizeof(T)));

    // build wavelet tree in-place
    buildWaveletTreeKernel<<<num_blocks, threads_per_block>>>(
        data, data_size, sorted_data, borders, num_levels, bit_array);
    kernelCheck();

    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_borders));

    // build rank and select structures from bit-vectors
    rank_select_(createRankSelectStructures(bit_array));
  }

  /*! Deleted copy constructor*/
  WaveletTree(const WaveletTree&) = delete;

  /*! Deleted copy assignment operator*/
  WaveletTree& operator=(const WaveletTree&) = delete;

  /*! Deleted move constructor*/
  WaveletTree(WaveletTree&&) = delete;

  /*! Deleted move assignment operator*/
  WaveletTree& operator=(WaveletTree&&) = delete;

  ~WaveletTree() {}

  /*!
   * \brief Access the symbols at the given indices in the wavelet tree.
   * \param indices Indices of the symbols to be accessed.
   * \return Vector of symbols.
   */
  __host__ std::vector<T> access(std::vector<size_t> const& indices) {
    // launch kernel with 1 warp per index
    size_t num_warps = indices.size();
    int num_blocks, int threads_per_block = getLaunchConfig(num_warps);

    // allocate space for results
    T* d_results;
    gpuErrchk(cudaMalloc(&d_results, num_warps * sizeof(T)));

    accessKernel<<<num_blocks, threads_per_block>>>(*this, indices.data(),
                                                    indices.size(), d_results);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // copy results back to host
    std::vector<T> results(num_warps);
    gpuErrchk(cudaMemcpy(results.data(), d_results, num_warps * sizeof(T),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_results));

#pragma omp parallel for
    for (size_t i = 0; i < num_warps; ++i) {
      results[i] = alphabet_[results[i]];
    }
    return results;
  }

  /*!
   * \brief Rank queries on the wavelet tree.
   * \param queries Vector of rank queries.
   * \return Vector of ranks.
   */
  __host__ std::vector<size_t> rank(
      std::vector<RankSelectQuery> const& queries) {
    // 1 warp per query
    // group characters together to reduce memory access
  }

  /*!
   * \brief Select queries on the wavelet tree.
   * \param queries Vector of select queries.
   * \return Vector of selected indices.
   */
  __host__ std::vector<size_t> select(
      std::vector<RankSelectQuery> const& queries) {
    // 1 warp per query
    // group characters together to reduce memory access
  }

  /*!
   * \brief Get the alphabet of the wavelet tree.
   * \return Vector of alphabet symbols.
   */
  template <typename T>
  __global__ void accessKernel(WaveletTree tree, size_t* const indices,
                               size_t const num_indices, T* results) {
    uint32_t glob_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WS;
    uint32_t local_t_id = threadIdx.x % WS;
    size_t index = indices[glob_warp_id];

    uint32_t char_start = 0;
    uint32_t char_end = tree.alphabet_size_;
    uint32_t start, pos;
    for (uint32_t l = 0; l < tree.num_levels_; ++l) {
      if (char_end - char_start == 1) {
        break;
      }
      start = tree.d_rank_select_structures_[l]->rank0(
          tree.d_counts_[char_start].count_, local_t_id, WS);
      pos = tree.d_rank_select_structures_[l]->rank0(
          tree.d_counts_[char_end].count_ + index, local_t_id, WS);
      if (tree.d_rank_select_structures_[l]->bit_array_.access(
              tree.d_counts_[char_start].count_ + index) == false) {
        index = pos - start;
        if (isPowTwo(char_end - char_start)) {
          char_end = char_start + (char_end - char_start) / 2;
        } else {
          char_end = char_start + getPrevPowTwo(char_end - char_start);
        }
      } else {
        i -= pos - start;
        if (isPowTwo(char_end - char_start)) {
          char_start += (char_end - char_start) / 2;
        } else {
          char_start += getPrevPowTwo(char_end - char_start);
        }
      }
    }
    results[glob_warp_id] = char_start;
  }

  __global__ void rankKernel(WaveletTree tree, RankSelectQuery* const queries,
                             size_t const num_queries, size_t* const ranks) {
    // for l = to to code length
  }

  __global__ void selectKernel(WaveletTree tree, RankSelectQuery* const queries,
                               size_t const num_queries, size_t* const ranks) {
    // for l = to to code length
  }

  /*!
   * \brief Encodes a symbol from the alphabet.
   * \param c Symbol to be encoded.
   * \return Encoded symbol.
   */
  __device__ Code encode(T& const c) {}

  /*!
   * \brief Get i-th most significant bit of a character.
   * \param i Index of the bit.
   * \param c Character to get the bit from.
   * \return Value of the bit.
   */
  __device__ bool getBit(uint8_t const i, T& const c) {
    assert(i < sizeof(T) * 8);
    return (c >> (sizeof(T) * 8 - i - 1)) & 1;
  }

 private:
  /*!
   * \brief Creates minimal codes for the alphabet.
   * \param alphabet Alphabet to create codes for.
   * \return Vector of codes.
   */
  __host__ std::vector<Code> createMinimalCodes(
      std::vector<T> const& alphabet) {
    size_t alphabet_size = alphabet.size();
    std::vector<Code> codes(alphabet_size);
    uint8_t total_num_bits = num_levels_;
#pragma omp parallel for
    for (size_t i = 0; i < alphabet_size; ++i) {
      codes[i].len_ = total_num_bits;
      codes[i].code_ = i;
    }
    uint8_t global_start_bit = 0;
    size_t global_start_i = 0;
    uint8_t code_len = total_num_bits;
    do {
      size_t num_codes = alphabet_size - global_start_i;
      uint8_t start_bit = global_start_bit;
      uint8_t num_bits = code_len;
      size_t start_i = global_start_i;

      for (uint32_t i = num_bits - 1; i > 0; --i) {
        if (num_codes <= powTwo<uint32_t>(i)) {
          break;
        }
        num_codes -= powTwo<uint32_t>(i);
        start_i += powTwo<uint32_t>(i);
        start_bit++;
      }
      if (num_codes == 1) {
        code_len = 1;
        codes[alphabet_size - 1].len_ = start_bit;
        codes[alphabet_size - 1].code_ = std::numeric_limits<T>::max()
                                         << (total_num_bits - start_bit + 1);
      } else {
        code_len = ceil(log2(num_codes));
#pragma omp parallel for
        for (int i = alphabet_size - num_codes; i < alphabet_size; i++) {
          // Code of local subtree
          T code = ((1UL << (total_num_bits - global_start_bit)) - 1) &
                   codes[i].code_;
          code -= (start_i - global_start_i);
          // Add to global code already saved
          code +=
              (~((1UL << (total_num_bits - start_bit)) - 1)) & codes[i].code_;

          codes[i].code_ = code;
          codes[i].len_ = start_bit + code_len;
        }
      }
      global_start_bit = start_bit;
      global_start_i = start_i;
    } while (code_len > 1);
    return codes;
  }

  /*!
   * \brief Struct for encoded alphabet caracter.
   */
  struct Code {
    size_t len_;
    T code_;
  };

  /*!
   * \brief Struct for extended version of pointerless wavelet tree.
   */
  //? Should not be necessary, since level can be inferred from code length
  struct Count {
    size_t count_; /*!< Number of symbols that are lexicographically smaller*/
    size_t level_; /*!< Level at which the symbol has it's leaf*/
  };

  RankSelect rank_select_;     /*!< Array of pointers to the rank
                                               and select data structures*/
  T* d_min_alphabet_;          //?Necessary?         /*!< Alphabet from
                               //[0..alphabet_size_)*/
  std::vector<T> alphabet_;    /*!< Alphabet of the wavelet tree*/
  size_t alphabet_size_;       /*!< Size of the alphabet*/
  uint8_t alphabet_start_bit_; /*!< Bit where the alphabet starts, 0 is the
                                  most significant bit*/
  Code* d_codes_;     /*!< Array of codes for each symbol in the alphabet*/
  size_t* d_counts_;  /*!< Array of counts for each symbol*/
  size_t num_levels_; /*!< Number of levels in the wavelet tree*/
};

template <typename T>
__global__ void computeGlobalHistogramKernel(T* data, size_t const data_size,
                                             size_t* counts, T* const alphabet,
                                             size_t const alphabet_size) {
  assert(blockDim.x % WS == 0);
  uint32_t total_threads = blockDim.x * gridDim.x;
  uint32_t global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = global_t_id; i < data_size; i += total_threads) {
    T char_data = data[i];
    size_t const char_index =
        lower_bound(alphabet, alphabet + alphabet_size, char_data) - alphabet;
    atomicAdd(&counts[char_index], 1);
    data[i] = char_index;
  }
}

template <typename T>
__global__ void buildLocalHistogramKernel(WaveletTree<T> tree, T* const data,
                                          size_t const data_size,
                                          size_t* histograms,
                                          size_t const entries_per_warp,
                                          uint8_t const alphabet_start_bit,
                                          BitArray bit_array) {
  assert(blockDim.x % WS == 0);
  uint32_t global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WS;
  uint8_t local_t_id = threadIdx.x % WS;
  uint32_t num_warps = gridDim.x * blockDim.x / WS;

  // size_t* warp_histogram = histograms + (global_warp_id)*entries_per_warp;
  // size_t* warp_borders = borders + (global_warp_id)*entries_per_warp;

  size_t const data_block_size = data_size / (blockDim.x * gridDim.x / WS);
  size_t const end = min(data_size, (global_warp_id + 1) * data_block_size);
  // Each warp processes a block of data
  for (uint32_t i = global_warp_id * data_block_size; i < end; i += WS) {
    T char_data = data[i + local_t_id];

    // Encode char
    Code const code = tree.encode(char_data);

    // Warp vote to all the bits that need to get written to a word
    uint32_t word = __ballot_sync(~0, getBit(alphabet_start_bit_, code.code_));

    if (threadIdx.x % WS == 0) {
      bit_array.writeWordAtBit(0, i, word);
    }

    //  Loop over code length and update local histogram
    size_t level_start = entries_per_warp - powTwo<size_t>(code.len_ - 1);
    size_t level_len = powTwo<size_t>(code.len_ - 1);

    //  Add to global histogram
    atomicAdd(&tree.d_counts_[char_data], 1);

    // TODO, probably one more, since last level not needed
    //  change char to have same length as code
    char_data >>= (sizeof(T) * 8 - code.len_);

    for (uint32_t l = code.len_ - 1; l > 0; --l) {
      atomicAdd(&histograms[level_start * num_warps +
                            global_warp_id * level_len + char_data],
                1);
      char_data >>= 1;
      level_len >>= 1;
      level_start -= level_len;
    }
  }
}

__global__ void communicateBordersKernel(size_t* borders,
                                         size_t* const d_local_histograms,
                                         size_t const entries_per_item,
                                         size_t const num_items) {
  size_t const global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_t_id < num_items * entries_per_item) {
    for (size_t i = 1; i < num_items; ++i) {
      borders[i * entries_per_item + global_t_id] =
          borders[(i - 1) * entries_per_item + global_t_id] +
          d_local_histograms[(i - 1) * entries_per_item + global_t_id];
    }
  }
}

template <typename T>
__global__ void buildWaveletTreeKernel(T* const data, size_t const data_size,
                                       T* sorted_data, size_t* borders,
                                       size_t const entries_per_warp,
                                       size_t const num_levels,
                                       BitArray bit_array) {
  assert(blockDim.x % WS == 0);
  __shared__ size_t write_pos[blockDim.x / WS];
  uint32_t const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WS;
  uint8_t const local_warp_id = threadIdx.x / WS;
  uint8_t const local_t_id = threadIdx.x % WS;
  uint32_t const num_warps = gridDim.x * blockDim.x / WS;

  size_t* warp_borders = borders + global_warp_id * entries_per_warp;

  size_t const data_block_size = data_size / (blockDim.x * gridDim.x / WS);
  size_t const start = global_warp_id * data_block_size;
  size_t const end = min(data_size, start + data_block_size);
  for (uint32_t l = 1; l < num_levels; ++l) {
    atomicExch(&write_pos[local_warp_id], 0);
    // Each warp processes a block of data
    for (uint32_t i = start; i < end; i += WS) {
      T char_data = data[i + local_t_id];

      Code const code = tree.encode(char_data);

      // TODO: Must happen in order, makes sense to use a warp?
      size_t const sorted_pos = warp_borders[bitPrefix(l, code.code_)]++;
      // TODO: when converting text, maybe convert to code instead of
      // min_alphabet
      sorted_data[sorted_pos] = code.code_;

      // reduce text
      if (code.len_ > l + 1) {
        // TODO: Must also happen in order
        size_t local_write_pos = atomicAdd(&write_pos[local_warp_id], 1);
        data[start + local_write_pos] = char_data;
      }
      __syncwarp();
      uint32_t const word =
          __ballot_sync(~0, getBit(l, sorted_data[i + local_t_id]));
      if (local_t_id % WS == 0) {
        bit_array.writeWordAtBit(l, i, word);
      }
    }
    __syncwarp();
    end = start + write_pos[local_warp_id];
  }
}
}  // namespace ecl