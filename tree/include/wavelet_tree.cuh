#pragma once

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

namespace ecl {

/*!
 * \brief Struct for creating a rank or select query.
 */
template <typename T>
struct RankSelectQuery {
  size_t index_;
  T symbol_;
};

/*!
 * \brief Helper class for reducing the data with thrust::remove_if.
 */
template <typename T>
class isNotLongEnough {
 public:
  isNotLongEnough(uint8_t* d_code_lens, uint32_t l, T codes_start)
      : d_code_lens_(d_code_lens), l_(l), codes_start_(codes_start) {}

  __device__ bool operator()(T const code) const {
    if (code < codes_start_) {
      return false;
    }
    assert(d_code_lens_[code - codes_start_] > 0);
    return d_code_lens_[code - codes_start_] <= l_ + 1;
  }

 private:
  uint8_t* d_code_lens_;
  uint32_t l_;
  T codes_start_;
};

/*!
 * \brief Wavelet tree class.
 * \tparam T Type of the text data the tree will be built upon.
 */
template <typename T>
class WaveletTree {
 public:
  /*!
   * \brief Struct for encoded alphabet caracter.
   */
  struct Code {
    uint8_t len_;
    T code_;
  };
  RankSelect
      rank_select_; /*!< RankSelect structure for constant time binary queries*/

  /*!
   * \brief Constructor. Builds the wavelet tree from the input data.
   * \param data Input data to build the wavelet tree.
   * \param data_size Number of elements in the input data.
   * \param alphabet Alphabet of the input data. Must be sorted.
   * \param GPU_index Index of the GPU to use.
   */
  __host__ WaveletTree(T* const data, size_t data_size,
                       std::vector<T>&& alphabet, uint8_t const GPU_index);

  /*! Copy constructor*/
  __host__ WaveletTree(WaveletTree const& other);

  /*! Deleted copy assignment operator*/
  WaveletTree& operator=(const WaveletTree&) = delete;

  /*! Deleted move constructor*/
  WaveletTree(WaveletTree&&) = delete;

  /*! Deleted move assignment operator*/
  WaveletTree& operator=(WaveletTree&&) = delete;

  /*! Destructor*/
  ~WaveletTree();

  /*!
   * \brief Access the symbols at the given indices in the wavelet tree.
   * \param indices Indices of the symbols to be accessed.
   * \return Vector of symbols.
   */
  __host__ std::vector<T> access(std::vector<size_t> const& indices);

  /*!
   * \brief Rank queries on the wavelet tree. Number of occurrences of a symbol
   * up to a given index (exclusive).
   * \param queries Vector of rank queries.
   * \return Vector of ranks.
   */
  __host__ std::vector<size_t> rank(std::vector<RankSelectQuery<T>>& queries);

  /*!
   * \brief Select queries on the wavelet tree. Find the index of the k-th
   * occurrence of a symbol. Starts counting from 1.
   * \param queries Vector of select queries.
   * \return Vector of selected indices.
   */
  __host__ std::vector<size_t> select(std::vector<RankSelectQuery<T>>& queries);

  /*!
   * \brief Encodes a symbol from the alphabet. Only symbols that have a code
   * should be passed as argument.
   * \param c Symbol of the minimal alphabet to be encoded.
   * \return Encoded symbol.
   */
  __device__ Code encode(T const c);

  /*!
   * \brief Creates minimal codes for the alphabet. Codes are only created for
   * the symbols that are bigger than the previous power of two of the alphabet
   * size.
   * \param alphabet Alphabet to create codes for.
   * \return Vector of codes.
   */
  __host__ static std::vector<Code> createMinimalCodes(
      std::vector<T> const& alphabet);

  /*!
   * \brief Get the size of the alphabet.
   * \return Size of the alphabet.
   */
  __device__ size_t getAlphabetSize() const;

  /*!
   * \brief Get the number of levels in the wavelet tree.
   * \return Number of levels.
   */
  __device__ size_t getNumLevels() const;

  /*!
   * \brief Get the number of occurrences of all symbols that are
   * lexicographically smaller than the i-th symbol in the alphabet. Starting
   * from 0. \param i Index of the symbol in the alphabet. \return Number of
   * occurrences of all symbols that are lexicographically smaller.
   */
  __device__ size_t getCounts(size_t i) const;

  /*!
   * \brief Return whether the alphabet is minimal.
   */
  __host__ __device__ bool isMinAlphabet() const;

  /*!
   * \brief Get the start of the codes in the alphabet.
   * \return Start of the codes.
   */
  __device__ T getCodesStart() const;

 protected:
  __host__ void computeGlobalHistogram(bool const is_pow_two,
                                       size_t const data_size, T* d_data,
                                       T* d_alphabet, size_t* d_histogram);

 private:
  std::vector<T> alphabet_;    /*!< Alphabet of the wavelet tree*/
  size_t alphabet_size_;       /*!< Size of the alphabet*/
  uint8_t alphabet_start_bit_; /*!< Bit where the alphabet starts, 0 is the
                                  least significant bit*/
  Code* d_codes_ =
      nullptr; /*!< Array of codes for each symbol in the alphabet*/
  uint8_t* d_code_lens_ =
      nullptr; /*!< Array of code lengths for each symbol in the alphabet*/
  size_t* d_counts_ = nullptr; /*!< Array of counts for each symbol*/
  size_t num_levels_;          /*!< Number of levels in the wavelet tree*/
  bool is_min_alphabet_; /*!< Flag to signal whether the alphabet is already
                            minimal*/
  T codes_start_;        /*!< Minimal alphabet symbol where the codes start*/
  bool is_copy_;         /*!< Flag to signal whether current object is a copy*/
};

/*!
 * \brief Kernel to compute the global histogram of the input data.
 * \details Also replaces the data with the codes instead of original symbols.
 * \param tree Wavelet tree.
 * \param data Input data.
 * \param data_size Number of elements in the input data.
 * \param counts Array to store the counts.
 * \param alphabet Alphabet of the input data.
 * \param alphabet_size Size of the alphabet.
 */
template <typename T, bool isMinAlphabet, bool isPowTwo, bool UseShmem>
__global__ void computeGlobalHistogramKernel(WaveletTree<T> tree, T* data,
                                             size_t const data_size,
                                             size_t* counts, T* const alphabet,
                                             size_t const alphabet_size,
                                             uint16_t const hists_per_block);

/*!
 * \brief Kernel to fill a level of the wavelet tree.
 * \param bit_array Bit array the level will be stored in.
 * \param data Data to be filled in the level.
 * \param data_size Size of the data.
 * \param alphabet_start_bit Bit where the alphabet starts. 0 is the LSB.
 * \param level Level to be filled.
 */
template <typename T>
__global__ void fillLevelKernel(BitArray bit_array, T* const data,
                                size_t const data_size,
                                uint8_t const alphabet_start_bit,
                                uint32_t const level);

/*!
 * \brief Kernel for computing access queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param indices Indices of the symbols to be accessed.
 * \param num_indices Number of indices.
 * \param results Array to store the accessed symbols.
 */
template <typename T>
__global__ void accessKernel(WaveletTree<T> tree, size_t* const indices,
                             size_t const num_indices, T* results);

/*!
 * \brief Kernel for computing rank queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param queries Array of rank queries.
 * \param num_queries Number of queries.
 * \param ranks Array to store the ranks.
 */
template <typename T>
__global__ void rankKernel(WaveletTree<T> tree,
                           RankSelectQuery<T>* const queries,
                           size_t const num_queries, size_t* const ranks);

/*!
 * \brief Kernel for computing select queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param queries Array of select queries.
 * \param num_queries Number of queries.
 * \param ranks Array to store the selected indices.
 */
template <typename T>
__global__ void selectKernel(WaveletTree<T> tree,
                             RankSelectQuery<T>* const queries,
                             size_t const num_queries, size_t* const ranks);

/****************************************************************************************
 * Implementation
 ****************************************************************************************/
typedef unsigned long long int cu_size_t;

template <typename T>
__host__ WaveletTree<T>::WaveletTree(T* const data, size_t data_size,
                                     std::vector<T>&& alphabet,
                                     uint8_t const GPU_index)
    : alphabet_(alphabet), is_copy_(false) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type");
  assert(data_size > 0);
  assert(alphabet_.size() > 0);
  assert(std::is_sorted(alphabet_.begin(), alphabet_.end()));

  //? Correct way?
  // Set device
  gpuErrchk(cudaSetDevice(GPU_index));
  checkWarpSize(GPU_index);
  alphabet_size_ = alphabet_.size();

  bool const is_pow_two = isPowTwo(alphabet_size_);

  // Check if alphabet is already minimal
  is_min_alphabet_ =
      std::all_of(alphabet_.begin(), alphabet_.end(),
                  [i = 0](unsigned value) mutable { return value == i++; });

  std::vector<Code> codes;
  // make minimal alphabet
  if (not is_min_alphabet_) {
    auto min_alphabet = std::vector<T>(alphabet_size_);
    std::iota(min_alphabet.begin(), min_alphabet.end(), 0);
    codes = createMinimalCodes(min_alphabet);
  } else {
    codes = createMinimalCodes(alphabet_);
  }

  num_levels_ = ceilLog2Host<T>(alphabet_size_);
  alphabet_start_bit_ = num_levels_ - 1;
  codes_start_ = alphabet_size_ - codes.size();

  // TODO separato codes from code lens
  //  create codes and copy to device
  if (codes.size() > 0) {
    gpuErrchk(cudaMalloc(&d_codes_, codes.size() * sizeof(Code)));
    gpuErrchk(cudaMemcpy(d_codes_, codes.data(), codes.size() * sizeof(Code),
                         cudaMemcpyHostToDevice));

    // TODO: Allocate as many code_lens_as codes, not more
    std::vector<uint8_t> code_lens(codes.back().code_ + 1 - codes_start_);
    for (size_t i = 0; i < alphabet_size_ - codes_start_; ++i) {
      code_lens[codes[i].code_ - codes_start_] = codes[i].len_;
    }
    gpuErrchk(cudaMalloc(&d_code_lens_, code_lens.size() * sizeof(uint8_t)));
    gpuErrchk(cudaMemcpy(d_code_lens_, code_lens.data(),
                         code_lens.size() * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));
  }

  // Allocate space for counts array
  gpuErrchk(cudaMalloc(&d_counts_, alphabet_size_ * sizeof(size_t)));
  gpuErrchk(cudaMemset(d_counts_, 0, alphabet_size_ * sizeof(size_t)));

  // Copy data to device
  T* d_data;
  gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
  gpuErrchk(
      cudaMemcpy(d_data, data, data_size * sizeof(T), cudaMemcpyHostToDevice));

  // Allocate space for sorted data
  T* d_sorted_data;
  gpuErrchk(cudaMalloc(&d_sorted_data, data_size * sizeof(T)));

  // Find necessary storage for CUB exclusive sum and radix sort
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts_,
                                d_counts_, alphabet_size_);

  size_t temp_storage_bytes_radix = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes_radix,
                                 d_data, d_sorted_data, data_size);

  temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_radix);
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  T* d_alphabet;
  if (alphabet_size_ * sizeof(T) <= temp_storage_bytes) {
    // Copy alphabet to device using temp_storage
    d_alphabet = reinterpret_cast<T*>(d_temp_storage);
    gpuErrchk(cudaMemcpy(d_temp_storage, alphabet_.data(),
                         alphabet_size_ * sizeof(T), cudaMemcpyHostToDevice));
  } else {
    gpuErrchk(cudaMalloc(&d_alphabet, alphabet_size_ * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_alphabet, alphabet_.data(),
                         alphabet_size_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  computeGlobalHistogram(is_pow_two, data_size, d_data, d_alphabet, d_counts_);

  // Copy counts to host
  std::vector<size_t> counts(alphabet_size_);
  gpuErrchk(cudaMemcpy(counts.data(), d_counts_,
                       alphabet_size_ * sizeof(size_t),
                       cudaMemcpyDeviceToHost));

  // Calculate size of bit array at each level
  std::vector<size_t> bit_array_sizes(num_levels_, data_size);
  if (codes.size() > 0) {  // Get min code length
    uint8_t const min_code_len = codes.back().len_;
#pragma omp parallel for
    for (size_t i = num_levels_ - 1; i >= min_code_len; --i) {
      for (int64_t j = alphabet_size_ - codes_start_ - 1; j >= 0; --j) {
        if (i >= codes[j].len_) {
          bit_array_sizes[i] -= counts[codes_start_ + j];
        } else {
          break;
        }
      }
    }
  }

  // Perform exclusive sum of counts
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts_,
                                d_counts_, alphabet_size_);

  BitArray bit_array(bit_array_sizes, false);

  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, fillLevelKernel<T>));
  uint32_t maxThreadsPerBlockFillLevel = funcAttrib.maxThreadsPerBlock;

  auto num_warps = (data_size + WS - 1) / WS;
  auto [num_blocks, threads_per_block] = getLaunchConfig(
      num_warps, kMinTPB, std::min(kMaxTPB, maxThreadsPerBlockFillLevel));

  fillLevelKernel<T><<<num_blocks, threads_per_block>>>(
      bit_array, d_data, data_size, alphabet_start_bit_, 0);
  kernelCheck();

  data_size = bit_array_sizes[1];
  for (uint32_t l = 1; l < num_levels_; l++) {
    assert(data_size == bit_array_sizes[l]);

    // Perform radix sort
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, d_data, d_sorted_data, data_size,
        alphabet_start_bit_ + 1 - l, alphabet_start_bit_ + 1);
    // TODO, could launch in different streams
    //  Fill l-th bit array
    num_warps = (data_size + WS - 1) / WS;
    std::tie(num_blocks, threads_per_block) = getLaunchConfig(
        num_warps, kMinTPB, std::min(kMaxTPB, maxThreadsPerBlockFillLevel));
    kernelCheck();

    fillLevelKernel<T><<<num_blocks, threads_per_block>>>(
        bit_array, d_sorted_data, data_size, alphabet_start_bit_, l);
    kernelCheck();

    if (l != (num_levels_ - 1) and
        bit_array_sizes[l] != bit_array_sizes[l + 1]) {
      //  Reduce text
      T* new_end =
          thrust::remove_if(thrust::device, d_data, d_data + data_size,
                            isNotLongEnough<T>(d_code_lens_, l, codes_start_));
      data_size = static_cast<size_t>(std::distance(d_data, new_end));
    }
  }

  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_sorted_data));
  if (alphabet_size_ * sizeof(T) > temp_storage_bytes) {
    gpuErrchk(cudaFree(d_alphabet));
  }
  gpuErrchk(cudaFree(d_temp_storage));

  // build rank and select structures from bit-vectors
  rank_select_ = RankSelect(std::move(bit_array), GPU_index);
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
      is_min_alphabet_(other.is_min_alphabet_),
      codes_start_(other.codes_start_),
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
      getLaunchConfig(num_indices, kMinTPB, maxThreadsPerBlockAccess);

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

  if (not is_min_alphabet_) {
#pragma omp parallel for
    for (size_t i = 0; i < num_indices; ++i) {
      results[i] = alphabet_[results[i]];
    }
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
      getLaunchConfig(num_queries, kMinTPB, maxThreadsPerBlock);

  // allocate space for results
  size_t* d_results;
  gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(size_t)));

  //  Convert query symbols to minimal alphabet
  if (not is_min_alphabet_) {
#pragma omp parallel for
    for (size_t i = 0; i < num_queries; ++i) {
      auto const symbol_index =
          std::lower_bound(alphabet_.begin(), alphabet_.end(),
                           queries[i].symbol_) -
          alphabet_.begin();
      assert(symbol_index < alphabet_size_);
      queries[i].symbol_ = static_cast<T>(symbol_index);
    }
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
      getLaunchConfig(num_queries, kMinTPB, maxThreadsPerBlock);

  // allocate space for results
  size_t* d_results;
  gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(size_t)));

  // TODO: Maybe convert to codes
  //  Convert query symbols to minimal alphabet
  if (not is_min_alphabet_) {
#pragma omp parallel for
    for (size_t i = 0; i < num_queries; ++i) {
      auto const symbol_index =
          std::lower_bound(alphabet_.begin(), alphabet_.end(),
                           queries[i].symbol_) -
          alphabet_.begin();
      assert(symbol_index < alphabet_size_);
      queries[i].symbol_ = static_cast<T>(symbol_index);
    }
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
  assert(c >= codes_start_);
  return d_codes_[c - codes_start_];
}

// TODO: Could be improved
template <typename T>
__host__ std::vector<typename WaveletTree<T>::Code>
WaveletTree<T>::createMinimalCodes(std::vector<T> const& alphabet) {
  auto const alphabet_size = alphabet.size();
  if (isPowTwo<size_t>(alphabet_size)) {
    return std::vector<Code>(0);
  }
  size_t total_num_codes = 0;
  std::vector<Code> codes(alphabet_size);
  uint8_t const total_num_bits = ceilLog2Host<T>(alphabet_size);
  uint8_t const alphabet_start_bit = total_num_bits - 1;
#pragma omp parallel for
  for (size_t i = 0; i < alphabet_size; ++i) {
    codes[i].len_ = total_num_bits;
    codes[i].code_ = i;
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
    // If its the first iteration
    if (code_len == total_num_bits) {
      total_num_codes = num_codes;
    }
    if (num_codes == 1) {
      code_len = 1;
      codes[alphabet_size - 1].len_ = start_bit;
      codes[alphabet_size - 1].code_ = ((1UL << start_bit) - 1)
                                       << (alphabet_start_bit + 1 - start_bit);
    } else {
      code_len = ceilLog2Host<T>(num_codes);
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
  // Keep only last alphabet_size - codes_start codes
  codes.erase(codes.begin(), codes.begin() + alphabet_size - total_num_codes);
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

template <typename T>
__host__ __device__ bool WaveletTree<T>::isMinAlphabet() const {
  return is_min_alphabet_;
}

template <typename T>
__device__ T WaveletTree<T>::getCodesStart() const {
  return codes_start_;
}

template <typename T>
__host__ void WaveletTree<T>::computeGlobalHistogram(bool const is_pow_two,
                                                     size_t const data_size,
                                                     T* d_data, T* d_alphabet,
                                                     size_t* d_histogram) {
  struct cudaFuncAttributes funcAttrib;
  if (is_pow_two) {
    if (is_min_alphabet_) {
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, computeGlobalHistogramKernel<T, true, true, true>));
    } else {
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, computeGlobalHistogramKernel<T, false, true, true>));
    }
  } else {
    if (is_min_alphabet_) {
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, computeGlobalHistogramKernel<T, true, false, true>));
    } else {
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, computeGlobalHistogramKernel<T, false, false, true>));
    }
  }

  int const maxThreadsPerBlockHist =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  struct cudaDeviceProp prop = getDeviceProperties();

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
  auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
  size_t const hist_size = sizeof(size_t) * alphabet_size_;

  auto const hists_per_SM = max_shmem_per_SM / hist_size;

  auto min_block_size =
      hists_per_SM < kMinBPM
          ? kMinTPB
          : std::max(kMinTPB,
                     static_cast<uint32_t>(max_threads_per_SM / hists_per_SM));

  // Make the minimum block size a multiple of WS
  min_block_size = ((min_block_size + WS - 1) / WS) * WS;
  // Compute global_histogram and change text to min_alphabet
  size_t num_warps = (data_size + WS - 1) / WS;
  if (hists_per_SM >= kMinBPM) {
    num_warps = std::min(
        num_warps,
        static_cast<size_t>(
            (max_threads_per_SM * prop.multiProcessorCount + WS - 1) / WS));
  }

  auto [num_blocks, threads_per_block] =
      getLaunchConfig(num_warps, min_block_size, maxThreadsPerBlockHist);

  uint16_t const blocks_per_SM = max_threads_per_SM / threads_per_block;

  size_t const used_shmem =
      std::min(max_shmem_per_SM / blocks_per_SM, prop.sharedMemPerBlock);

  uint16_t const hists_per_block =
      std::min(static_cast<size_t>(threads_per_block), used_shmem / hist_size);

  if (is_pow_two) {
    if (is_min_alphabet_) {
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, true, true, true>
            <<<num_blocks, threads_per_block, used_shmem>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, true, true, false>
            <<<num_blocks, threads_per_block>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      }
    } else {
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, false, true, true>
            <<<num_blocks, threads_per_block, used_shmem>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, false, true, false>
            <<<num_blocks, threads_per_block>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      }
    }
  } else {
    if (is_min_alphabet_) {
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, true, false, true>
            <<<num_blocks, threads_per_block, used_shmem>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, true, false, false>
            <<<num_blocks, threads_per_block>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      }
    } else {
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, false, false, true>
            <<<num_blocks, threads_per_block, used_shmem>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, false, false, false>
            <<<num_blocks, threads_per_block>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      }
    }
  }
  kernelCheck();
}

template <typename T, bool isMinAlphabet, bool isPowTwo, bool UseShmem>
__global__ LB(MAX_TPB, MIN_BPM) void computeGlobalHistogramKernel(
    WaveletTree<T> tree, T* data, size_t const data_size, size_t* counts,
    T* const alphabet, size_t const alphabet_size,
    uint16_t const hists_per_block) {
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t shared_hist[];
  size_t offset;
  if constexpr (UseShmem) {
    offset = (threadIdx.x % hists_per_block) * alphabet_size;
    for (size_t i = threadIdx.x; i < alphabet_size * hists_per_block;
         i += blockDim.x) {
      shared_hist[i] = 0;
    }
    __syncthreads();
  }

  size_t const total_threads = blockDim.x * gridDim.x;
  size_t const global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
  T char_data;
  for (size_t i = global_t_id; i < data_size; i += total_threads) {
    char_data = data[i];
    if constexpr (not isMinAlphabet) {
      char_data = thrust::lower_bound(thrust::seq, alphabet,
                                      alphabet + alphabet_size, char_data) -
                  alphabet;
    }
    // TODO: atomic block not available for CC <=5.2
    if constexpr (UseShmem) {
      atomicAdd_block((cu_size_t*)&shared_hist[offset + char_data], size_t(1));
    } else {
      atomicAdd((cu_size_t*)&counts[char_data], size_t(1));
    }
    // TODO: could calculate code on the fly
    if constexpr (not isPowTwo) {
      if (char_data >= tree.getCodesStart()) {
        char_data = tree.encode(char_data).code_;
      }
    }
    data[i] = char_data;
  }

  if constexpr (UseShmem) {
    __syncthreads();
    // Reduce shared histograms to first one
    // TODO: Could maybe be improved
    for (size_t i = threadIdx.x; i < alphabet_size; i += blockDim.x) {
      size_t sum = shared_hist[i];
      for (size_t j = 1; j < hists_per_block; ++j) {
        sum += shared_hist[j * alphabet_size + i];
      }
      atomicAdd((cu_size_t*)&counts[i], sum);
    }
  }
}

template <typename T>
__global__ void fillLevelKernel(BitArray bit_array, T* const data,
                                size_t const data_size,
                                uint8_t const alphabet_start_bit,
                                uint32_t const level) {
  assert(blockDim.x % WS == 0);
  size_t const global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t const num_threads = gridDim.x * blockDim.x;
  uint8_t const local_t_id = threadIdx.x % WS;

  size_t const data_size_rounded = (data_size + (WS - 1)) & ~(WS - 1);
  size_t const offset = bit_array.getOffset(level);

  // Each warp processes a block of data
  for (size_t i = global_t_id; i < data_size_rounded; i += num_threads) {
    T code = 0;
    if (i < data_size) {
      code = data[i];
    }

    // Warp vote to all the bits that need to get written to a word
    uint32_t word = __ballot_sync(~0, getBit(alphabet_start_bit - level, code));

    if (local_t_id == 0) {
      bit_array.writeWordAtBit(level, i, word, offset);
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
      // TODO: could be done in parallel, and combined if index is less than
      // L2 block size
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
      // TODO: could be done in parallel, and combined if index is less than
      // L2 block size
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
  auto const codes_start = tree.getCodesStart();

  for (uint32_t i = global_warp_id; i < num_queries; i += num_warps) {
    RankSelectQuery<T> query = queries[i];
    typename WaveletTree<T>::Code code{alphabet_num_bits, query.symbol_};
    if (query.symbol_ >= codes_start) {
      code = tree.encode(query.symbol_);
    }

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
      results[i] = query.index_ - 1;  // 0-indexed
    }
  }
}
}  // namespace ecl
