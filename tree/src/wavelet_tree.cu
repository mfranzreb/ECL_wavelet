#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <algorithm>
#include <cmath>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <numeric>
#include <vector>

#include "rank_select.cuh"
#include "utils.cuh"
#include "wavelet_tree.cuh"

// TODO: set max block size of kernels to max allowed by kernel properties

namespace ecl {
__device__ size_t d_data_len;
typedef unsigned long long int cu_size_t;

// TODO: How to circumvent this
template class WaveletTree<uint8_t>;
template class WaveletTree<uint16_t>;
template class WaveletTree<uint32_t>;
template class WaveletTree<uint64_t>;

template <typename T>
__host__ WaveletTree<T>::WaveletTree(T* const data, size_t data_size,
                                     std::vector<T>&& alphabet,
                                     uint8_t const GPU_index)
    : alphabet_(alphabet), is_copy_(false) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type");
  assert(data_size > 0);
  assert(alphabet.size() > 0);
  assert(std::is_sorted(alphabet.begin(), alphabet.end()));

  // Set device
  gpuErrchk(cudaSetDevice(GPU_index));
  checkWarpSize();
  // make minimal alphabet
  alphabet_size_ = alphabet.size();
  std::vector<T> min_alphabet(alphabet_size_);
  std::iota(min_alphabet.begin(), min_alphabet.end(), 0);

  num_levels_ = ceil(log2(alphabet_size_));
  alphabet_start_bit_ = num_levels_ - 1;

  // TODO separato codes from code lens
  //  create codes and copy to device
  std::vector<Code> codes = createMinimalCodes(min_alphabet);
  gpuErrchk(cudaMalloc(&d_codes_, alphabet_size_ * sizeof(Code)));
  gpuErrchk(cudaMemcpy(d_codes_, codes.data(), alphabet_size_ * sizeof(Code),
                       cudaMemcpyHostToDevice));

  std::vector<uint8_t> code_lens(codes.back().code_ + 1);
  for (size_t i = 0; i < alphabet_size_; ++i) {
    code_lens[codes[i].code_] = codes[i].len_;
  }
  gpuErrchk(cudaMalloc(&d_code_lens_, code_lens.size() * sizeof(uint8_t)));
  gpuErrchk(cudaMemcpy(d_code_lens_, code_lens.data(),
                       code_lens.size() * sizeof(uint8_t),
                       cudaMemcpyHostToDevice));

  // Allocate space for counts array
  gpuErrchk(cudaMalloc(&d_counts_, alphabet_size_ * sizeof(size_t)));
  gpuErrchk(cudaMemset(d_counts_, 0, alphabet_size_ * sizeof(size_t)));

  // Copy data to device
  T* d_data;
  gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
  gpuErrchk(
      cudaMemcpy(d_data, data, data_size * sizeof(T), cudaMemcpyHostToDevice));

  // Copy alphabet to device
  T* d_alphabet;
  gpuErrchk(cudaMalloc(&d_alphabet, alphabet_size_ * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_alphabet, alphabet.data(), alphabet_size_ * sizeof(T),
                       cudaMemcpyHostToDevice));

  // Compute global_histogram and change text to min_alphabet
  // TODO: find appropriate number of warps, as many as possible
  auto num_warps = std::min((data_size + WS - 1) / WS, 1'000'000UL);
  auto [num_blocks, threads_per_block] =
      getLaunchConfig(num_warps, 256, MAX_TPB);
  computeGlobalHistogramKernel<T><<<num_blocks, threads_per_block>>>(
      *this, d_data, data_size, d_counts_, d_alphabet, alphabet_size_);

  // Copy counts to host
  std::vector<size_t> counts(alphabet_size_);
  gpuErrchk(cudaMemcpy(counts.data(), d_counts_,
                       alphabet_size_ * sizeof(size_t),
                       cudaMemcpyDeviceToHost));

  // Calculate size of bit array at each level
  std::vector<size_t> bit_array_sizes(num_levels_, data_size);
  // Get min code length
  uint8_t const min_code_len = codes.back().len_;
#pragma omp parallel for
  for (size_t i = num_levels_ - 1; i >= min_code_len; --i) {
    for (int64_t j = alphabet_size_ - 1; j >= 0; --j) {
      if (i >= codes[j].len_) {
        bit_array_sizes[i] -= counts[j];
      } else {
        break;
      }
    }
  }

  // Perform exclusive sum of counts
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_counts_,
                                d_counts_, alphabet_size_);

  // TODO use same memory from RadixSort
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts_,
                                d_counts_, alphabet_size_);
  gpuErrchk(cudaFree(d_temp_storage));

  BitArray bit_array(bit_array_sizes, false);

  gpuErrchk(cudaMemcpyToSymbol(d_data_len, &data_size, sizeof(size_t),
                               size_t(0), cudaMemcpyHostToDevice));
  fillLevelKernel<T><<<num_blocks, threads_per_block>>>(bit_array, d_data,
                                                        alphabet_start_bit_, 0);

  // Allocate space for sorted data
  T* d_sorted_data;
  gpuErrchk(cudaMalloc(&d_sorted_data, data_size * sizeof(T)));

  d_temp_storage = nullptr;
  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data,
                                 d_sorted_data, data_size);
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  data_size = bit_array_sizes[1];
  for (uint32_t l = 1; l < num_levels_; l++) {
    assert(data_size == bit_array_sizes[l]);
    gpuErrchk(cudaMemcpyToSymbol(d_data_len, &data_size, sizeof(size_t),
                                 size_t(0), cudaMemcpyHostToDevice));
    // Perform radix sort
    // TODO: look at different versions
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, d_data, d_sorted_data, data_size,
        alphabet_start_bit_ + 1 - l, alphabet_start_bit_ + 1);
    // TODO, could launch in different streams
    //  Fill l-th bit array
    fillLevelKernel<T><<<num_blocks, threads_per_block>>>(
        bit_array, d_sorted_data, alphabet_start_bit_, l);
    kernelCheck();

    if (l != (num_levels_ - 1) and
        bit_array_sizes[l] != bit_array_sizes[l + 1]) {
      //  Reduce text
      T* new_end = thrust::remove_if(thrust::device, d_data, d_data + data_size,
                                     isLongEnough<T>(d_code_lens_, l));
      data_size = static_cast<size_t>(std::distance(d_data, new_end));
    }
  }

  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_sorted_data));
  gpuErrchk(cudaFree(d_alphabet));
  gpuErrchk(cudaFree(d_temp_storage));

  // build rank and select structures from bit-vectors
  rank_select_ = RankSelect(std::move(bit_array));
}

template <typename T>
__host__ WaveletTree<T>::WaveletTree(WaveletTree const& other)
    : alphabet_(other.alphabet_),
      rank_select_(other.rank_select_),
      alphabet_size_(other.alphabet_size_),
      alphabet_start_bit_(other.alphabet_start_bit_),
      num_levels_(other.num_levels_),
      d_codes_(other.d_codes_),
      d_code_lens_(other.d_code_lens_),
      d_counts_(other.d_counts_),
      is_copy_(true) {}

template <typename T>
WaveletTree<T>::~WaveletTree() {
  if (not is_copy_) {
    gpuErrchk(cudaFree(d_codes_));
    gpuErrchk(cudaFree(d_code_lens_));
    gpuErrchk(cudaFree(d_counts_));
  }
}

template <typename T>
__host__ std::vector<T> WaveletTree<T>::access(
    std::vector<size_t> const& indices) {
  // launch kernel with 1 warp per index
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, accessKernel<T>));
  int maxThreadsPerBlockAccess = funcAttrib.maxThreadsPerBlock;
  size_t num_indices = indices.size();
  auto [num_blocks, threads_per_block] =
      getLaunchConfig(num_indices, 256, maxThreadsPerBlockAccess);

  // allocate space for results
  T* d_results;
  gpuErrchk(cudaMalloc(&d_results, num_indices * sizeof(T)));

  // Copy indices to device
  size_t* d_indices;
  gpuErrchk(cudaMalloc(&d_indices, num_indices * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_indices, indices.data(), num_indices * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  accessKernel<T><<<num_blocks, threads_per_block>>>(*this, d_indices,
                                                     num_indices, d_results);
  kernelCheck();

  // copy results back to host
  std::vector<T> results(num_indices);
  gpuErrchk(cudaMemcpy(results.data(), d_results, num_indices * sizeof(T),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_indices));
  gpuErrchk(cudaFree(d_results));

#pragma omp parallel for
  for (size_t i = 0; i < num_indices; ++i) {
    results[i] = alphabet_[results[i]];
  }
  return results;
}

template <typename T>
__host__ std::vector<size_t> WaveletTree<T>::rank(
    std::vector<RankSelectQuery<T>>& queries) {
  assert(std::all_of(queries.begin(), queries.end(),
                     [&](const RankSelectQuery<T>& s) {
                       return s.index_ < rank_select_.bit_array_.sizeHost(0);
                     }));
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, rankKernel<T>));
  int maxThreadsPerBlock = funcAttrib.maxThreadsPerBlock;
  // launch kernel with 1 warp per index
  size_t const num_queries = queries.size();
  auto [num_blocks, threads_per_block] =
      getLaunchConfig(num_queries, 256, maxThreadsPerBlock);

  // allocate space for results
  size_t* d_results;
  gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(size_t)));

  // TODO: have flag to check if alphabet is already minimal in constructor
  //  Convert query symbols to minimal alphabet
#pragma omp parallel for
  for (size_t i = 0; i < num_queries; ++i) {
    auto const symbol_index =
        std::lower_bound(alphabet_.begin(), alphabet_.end(),
                         queries[i].symbol_) -
        alphabet_.begin();
    assert(symbol_index < alphabet_size_);
    queries[i].symbol_ = static_cast<T>(symbol_index);
  }

  // Copy queries to device
  RankSelectQuery<T>* d_queries;
  gpuErrchk(cudaMalloc(&d_queries, num_queries * sizeof(RankSelectQuery<T>)));
  gpuErrchk(cudaMemcpy(d_queries, queries.data(),
                       num_queries * sizeof(RankSelectQuery<T>),
                       cudaMemcpyHostToDevice));

  rankKernel<T><<<num_blocks, threads_per_block>>>(*this, d_queries,
                                                   num_queries, d_results);
  kernelCheck();

  // copy results back to host
  std::vector<size_t> results(num_queries);
  gpuErrchk(cudaMemcpy(results.data(), d_results, num_queries * sizeof(size_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_queries));
  gpuErrchk(cudaFree(d_results));

  return results;
}

template <typename T>
__host__ std::vector<size_t> WaveletTree<T>::select(
    std::vector<RankSelectQuery<T>>& queries) {
  assert(std::all_of(queries.begin(), queries.end(),
                     [](const RankSelectQuery<T>& s) { return s.index_ > 0; }));
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, selectKernel<T>));
  int maxThreadsPerBlock = funcAttrib.maxThreadsPerBlock;
  // launch kernel with 1 warp per index
  size_t const num_queries = queries.size();
  auto [num_blocks, threads_per_block] =
      getLaunchConfig(num_queries, 256, maxThreadsPerBlock);

  // allocate space for results
  size_t* d_results;
  gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(size_t)));

  // Convert query symbols to minimal alphabet
#pragma omp parallel for
  for (size_t i = 0; i < num_queries; ++i) {
    auto const symbol_index =
        std::lower_bound(alphabet_.begin(), alphabet_.end(),
                         queries[i].symbol_) -
        alphabet_.begin();
    assert(symbol_index < alphabet_size_);
    queries[i].symbol_ = static_cast<T>(symbol_index);
  }

  // Copy queries to device
  RankSelectQuery<T>* d_queries;
  gpuErrchk(cudaMalloc(&d_queries, num_queries * sizeof(RankSelectQuery<T>)));
  gpuErrchk(cudaMemcpy(d_queries, queries.data(),
                       num_queries * sizeof(RankSelectQuery<T>),
                       cudaMemcpyHostToDevice));

  selectKernel<T><<<num_blocks, threads_per_block>>>(*this, d_queries,
                                                     num_queries, d_results);
  kernelCheck();

  // copy results back to host
  std::vector<size_t> results(num_queries);
  gpuErrchk(cudaMemcpy(results.data(), d_results, num_queries * sizeof(size_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_queries));
  gpuErrchk(cudaFree(d_results));

  return results;
}

template <typename T>
__device__ WaveletTree<T>::Code WaveletTree<T>::encode(T const c) {
  assert(c < alphabet_size_);
  return d_codes_[c];
}

template <typename T>
__host__ std::vector<typename WaveletTree<T>::Code>
WaveletTree<T>::createMinimalCodes(std::vector<T> const& alphabet) {
  auto const alphabet_size = alphabet.size();
  std::vector<Code> codes(alphabet_size);
  uint8_t const total_num_bits = ceil(log2(alphabet_size));
  uint8_t const alphabet_start_bit = total_num_bits - 1;
#pragma omp parallel for
  for (size_t i = 0; i < alphabet_size; ++i) {
    codes[i].len_ = total_num_bits;
    codes[i].code_ = i;
  }
  if (isPowTwo<size_t>(alphabet_size)) {
    return codes;
  }
  uint8_t start_bit = 0;  // 0 is the alphabet start bit.
  size_t start_i = 0;
  uint8_t code_len = total_num_bits;
  size_t num_codes = alphabet_size;
  do {
    for (uint32_t i = code_len - 1; i > 0; --i) {
      auto pow_two = powTwo<uint32_t>(i);
      if (num_codes <= pow_two) {
        break;
      }
      num_codes -= pow_two;
      start_i += pow_two;
      start_bit++;
    }
    if (num_codes == 1) {
      code_len = 1;
      codes[alphabet_size - 1].len_ = start_bit;
      codes[alphabet_size - 1].code_ = ((1UL << start_bit) - 1)
                                       << (alphabet_start_bit + 1 - start_bit);
    } else {
      code_len = ceil(log2(num_codes));
#pragma omp parallel for
      for (int i = alphabet_size - num_codes; i < alphabet_size; i++) {
        // Code of local subtree
        T code = i - start_i;
        // Shift code to start at start_bit
        code <<= (total_num_bits - start_bit - code_len);
        // Add to global code already saved
        code += (~((1UL << (total_num_bits - start_bit)) - 1)) & codes[i].code_;

        codes[i].code_ = code;
        codes[i].len_ = start_bit + code_len;
      }
    }
  } while (code_len > 1);
  return codes;
}

template <typename T>
__device__ size_t WaveletTree<T>::getAlphabetSize() const {
  return alphabet_size_;
}

template <typename T>
__device__ size_t WaveletTree<T>::getNumLevels() const {
  return num_levels_;
}

template <typename T>
__device__ size_t WaveletTree<T>::getCounts(size_t i) const {
  return d_counts_[i];
}

// local block hist with shmem
template <typename T>
__global__ void computeGlobalHistogramKernel(WaveletTree<T> tree, T* data,
                                             size_t const data_size,
                                             size_t* counts, T* const alphabet,
                                             size_t const alphabet_size) {
  assert(blockDim.x % WS == 0);
  uint32_t total_threads = blockDim.x * gridDim.x;
  uint32_t global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = global_t_id; i < data_size; i += total_threads) {
    T const char_data = data[i];
    // TODO: avoid if alphabet already minimal
    size_t const char_index =
        thrust::lower_bound(thrust::seq, alphabet, alphabet + alphabet_size,
                            char_data) -
        alphabet;
    typename WaveletTree<T>::Code const code = tree.encode(char_index);
    atomicAdd((cu_size_t*)&counts[char_index], size_t(1));
    data[i] = code.code_;
  }
}

template <typename T>
__global__ void fillLevelKernel(BitArray bit_array, T* const data,
                                uint8_t const alphabet_start_bit,
                                uint32_t const level) {
  assert(blockDim.x % WS == 0);
  uint32_t const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WS;
  uint32_t const num_warps = gridDim.x * blockDim.x / WS;
  uint8_t const local_t_id = threadIdx.x % WS;

  size_t const start = WS * global_warp_id;
  // Each warp processes a block of data
  for (uint32_t i = start; i < d_data_len; i += WS * num_warps) {
    T code = 0;
    // TODO: remove if
    if (i + local_t_id >= d_data_len) {
      code = 0;
    } else {
      code = data[i + local_t_id];
    }

    // Warp vote to all the bits that need to get written to a word
    uint32_t word = __ballot_sync(~0, getBit(alphabet_start_bit - level, code));

    if (local_t_id == 0) {
      bit_array.writeWordAtBit(level, i, word);
    }
  }
}

template <typename T>
__global__ void accessKernel(WaveletTree<T> tree, size_t* const indices,
                             size_t const num_indices, T* results) {
  assert(blockDim.x % WS == 0);
  uint32_t const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WS;
  uint32_t const num_warps = gridDim.x * blockDim.x / WS;
  uint32_t const local_t_id = threadIdx.x % WS;

  for (uint32_t i = global_warp_id; i < num_indices; i += num_warps) {
    size_t index = indices[i];

    uint32_t char_start = 0;
    uint32_t char_end = tree.getAlphabetSize();
    uint32_t start, pos;
    for (uint32_t l = 0; l < tree.getNumLevels(); ++l) {
      if (char_end - char_start == 1) {
        break;
      }
      // TODO: could be done in parallel, and combined if index is less than L2
      // block size
      start = tree.rank_select_.rank0(l, tree.getCounts(char_start), local_t_id,
                                      WS);
      pos = tree.rank_select_.rank0(l, tree.getCounts(char_start) + index,
                                    local_t_id, WS);
      if (tree.rank_select_.bit_array_.access(
              l, tree.getCounts(char_start) + index) == false) {
        index = pos - start;
        char_end = char_start + getPrevPowTwo(char_end - char_start);
      } else {
        index -= pos - start;
        char_start += getPrevPowTwo(char_end - char_start);
      }
    }
    if (local_t_id == 0) {
      results[i] = char_start;
    }
  }
}

template <typename T>
__global__ void rankKernel(WaveletTree<T> tree,
                           RankSelectQuery<T>* const queries,
                           size_t const num_queries, size_t* const ranks) {
  assert(blockDim.x % WS == 0);
  uint32_t const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WS;
  uint32_t const num_warps = gridDim.x * blockDim.x / WS;
  uint32_t const local_t_id = threadIdx.x % WS;

  for (uint32_t i = global_warp_id; i < num_queries; i += num_warps) {
    RankSelectQuery<T> query = queries[i];

    uint32_t char_start = 0;
    uint32_t char_end = tree.getAlphabetSize();
    uint32_t char_split;
    size_t start, pos;
    for (uint32_t l = 0; l < tree.getNumLevels(); ++l) {
      if (char_end - char_start == 1) {
        break;
      }
      // TODO: could be done in parallel, and combined if index is less than L2
      // block size
      start = tree.rank_select_.rank0(l, tree.getCounts(char_start), local_t_id,
                                      WS);
      pos = tree.rank_select_.rank0(
          l, tree.getCounts(char_start) + query.index_, local_t_id, WS);
      char_split = char_start + getPrevPowTwo(char_end - char_start);
      if (query.symbol_ < char_split) {
        query.index_ = pos - start;
        char_end = char_split;
      } else {
        query.index_ -= pos - start;
        char_start = char_split;
      }
    }
    if (local_t_id == 0) {
      ranks[i] = query.index_;
    }
  }
}

__device__ uint32_t getPrevCharStart(uint32_t const char_start,
                                     bool const is_rightmost_child,
                                     uint32_t const alphabet_size,
                                     uint32_t const level,
                                     uint8_t const code_len) {
  if (is_rightmost_child) {
    uint32_t new_start = 0;
    // TODO: look if lookup table is faster
    for (uint32_t l = 0; l < level; l++) {
      new_start += getPrevPowTwo(alphabet_size - new_start);
    }
    return new_start;
  } else {
    return char_start - powTwo<uint32_t>(code_len - 1 - level);
  }
}

template <typename T>
__global__ void selectKernel(WaveletTree<T> tree,
                             RankSelectQuery<T>* const queries,
                             size_t const num_queries, size_t* const results) {
  assert(blockDim.x % WS == 0);
  uint32_t const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WS;
  uint32_t const num_warps = gridDim.x * blockDim.x / WS;
  uint32_t const local_t_id = threadIdx.x % WS;

  uint32_t const alphabet_size = tree.getAlphabetSize();
  uint8_t const alphabet_num_bits = ceilLog2<uint32_t>(alphabet_size);

  for (uint32_t i = global_warp_id; i < num_queries; i += num_warps) {
    RankSelectQuery<T> query = queries[i];
    typename WaveletTree<T>::Code const code = tree.encode(query.symbol_);

    uint32_t char_start = query.symbol_;
    size_t start;
    for (int32_t l = code.len_ - 1; l >= 0; --l) {
      // If it's a right child
      if (getBit<T>(alphabet_num_bits - 1 - l, code.code_) == true) {
        // Is rightmost child if bit-prefix of the code is all 1s
        bool is_rightmost_child =
            __popc(code.code_ >> (alphabet_num_bits - l)) == l;

        char_start = getPrevCharStart(char_start, is_rightmost_child,
                                      alphabet_size, l, code.len_);
        start = tree.rank_select_.rank1(l, tree.getCounts(char_start),
                                        local_t_id, WS);
        query.index_ = tree.rank_select_.select<1>(l, start + query.index_,
                                                   local_t_id, WS) +
                       1;
      } else {
        start = tree.rank_select_.rank0(l, tree.getCounts(char_start),
                                        local_t_id, WS);
        query.index_ = tree.rank_select_.select<0>(l, start + query.index_,
                                                   local_t_id, WS) +
                       1;
      }
      if (l == (code.len_ - 1) and
          query.index_ > tree.rank_select_.bit_array_.size(l)) {
        query.index_ = tree.rank_select_.bit_array_.size(0) + 1;
        break;
      }
      query.index_ -= tree.getCounts(char_start);
    }
    if (local_t_id == 0) {
      results[i] = query.index_ - 1;  // 0-indexed}
    }
  }
}
}  // namespace ecl