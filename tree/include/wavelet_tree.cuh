#pragma once

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <algorithm>
#include <cmath>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <numeric>
#include <thread>
#include <vector>

#include "rank_select.cuh"
#include "utils.cuh"

// TODO: add restricts
namespace ecl {
__device__ size_t access_counter = 0;
__device__ size_t rank_counter = 0;
struct GraphContents {
  cudaGraphExec_t graph_exec;
  std::vector<cudaGraphNode_t> copy_indices_nodes;
  std::vector<cudaGraphNode_t> kernel_nodes;
  std::vector<cudaGraphNode_t> copy_results_nodes;
};

template <typename T>
struct NodeInfo {
  T start_;
  uint8_t level_;
};
std::unordered_map<uint16_t, GraphContents> queries_graph_cache;

/*!
 * \brief Struct for creating a rank or select query.
 */
template <typename T>
struct RankSelectQuery;

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
   * \brief Access the symbols at the given indices in the wavelet tree. The
   * indices array must be allocated with pinned memory.
   * \param indices Pointer to the indices of the symbols to be accessed.
   * \param num_indices Number of indices.
   * \return Vector of symbols.
   */
  template <int NumThreads = WS>
  __host__ std::span<T> access(size_t* indices, size_t const num_indices);

  /*!
   * \brief Rank queries on the wavelet tree. Number of occurrences of a symbol
   * up to a given index (exclusive).
   * \param queries Vector of rank queries.
   * \return Vector of ranks.
   */
  template <int NumThreads = WS>
  __host__ std::span<size_t> rank(RankSelectQuery<T>* queries,
                                  size_t const num_queries);

  /*!
   * \brief Select queries on the wavelet tree. Find the index of the k-th
   * occurrence of a symbol. Starts counting from 1.
   * \param queries Vector of select queries.
   * \return Vector of selected indices.
   */
  template <int ThreadsPerQuery = WS>
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
  __device__ uint8_t getNumLevels() const;

  /*!
   * \brief Get the number of occurrences of all symbols that are
   * lexicographically smaller than the i-th symbol in the alphabet. Starting
   * from 0. \param i Index of the symbol in the alphabet. \return Number of
   * occurrences of all symbols that are lexicographically smaller.
   */
  __device__ size_t getCounts(size_t i) const;

  __device__ size_t getTotalAppearances(size_t i) const noexcept {
    if (i == alphabet_size_ - 1) {
      return rank_select_.bit_array_.size(0) -
             getCounts(i);  // data_size - getCounts(i);
    } else {
      return getCounts(i + 1) - getCounts(i);
    }
  }

  /*!
   * \brief Return whether the alphabet is minimal.
   */
  __host__ __device__ bool isMinAlphabet() const;

  /*!
   * \brief Get the start of the codes in the alphabet.
   * \return Start of the codes.
   */
  __device__ T getCodesStart() const;

  template <bool IsPowTwo>
  __device__ T getNodePosAtLevel(T symbol, uint8_t const level) const {
    if (level == 1) {
      return 0;
    }
    T num_prev_nodes = 0;
    T node_lens = 1;
    if constexpr (IsPowTwo) {
      // All node lens are equal
      node_lens = 1ULL << (num_levels_ - level);
    } else {
      T remaining_symbols = alphabet_size_;
      for (int16_t i = level - 1; i >= 0; --i) {
        auto const subtree_size = getPrevPowTwo<T>(remaining_symbols);
        node_lens = subtree_size >> i;
        if (symbol <= subtree_size) {
          break;
        } else {
          remaining_symbols -= subtree_size;
          num_prev_nodes += subtree_size / node_lens;
          symbol -= subtree_size;
        }
      }
    }
    return symbol / node_lens + num_prev_nodes - 1;  // First node doesnt count
  }

  __device__ size_t getPrecomputedRank(T const i) const { return d_ranks_[i]; }

  __device__ void setPrecomputedRank(T const i, size_t const rank) {
    d_ranks_[i] = rank;
  }

  __device__ T getNumNodesAtLevel(uint8_t const level) const {
    return d_num_nodes_at_level_[level];
  }

 protected:
  __host__ void computeGlobalHistogram(bool const is_pow_two,
                                       size_t const data_size, T* d_data,
                                       T* d_alphabet, size_t* d_histogram);

  __host__ void fillLevel(BitArray bit_array, T* const data,
                          size_t const data_size, uint32_t const level);

  __host__ static std::vector<NodeInfo<T>> getNodeInfos(
      std::vector<T> const& alphabet, std::vector<Code> const& codes) {
    auto const alphabet_size = alphabet.size();
    auto const symbol_len = static_cast<uint8_t>(ceilLog2Host(alphabet_size));

    std::vector<Code> alphabet_codes(alphabet_size);
    for (T i = 0; i < alphabet_size; ++i) {
      if (i < alphabet_size - codes.size()) {
        alphabet_codes[i] = {symbol_len, i};
      } else {
        alphabet_codes[i] = codes[i - (alphabet_size - codes.size())];
      }
    }

    std::vector<NodeInfo<T>> node_starts;
    for (uint8_t l = 1; l < symbol_len; ++l) {
      // Find each position where the l-th MSB of the symbol is 0 and the
      // previous 1
      for (size_t i = 1; i < alphabet_size; ++i) {
        if (alphabet_codes[i].len_ > l and
            (getBit(symbol_len - l - 1, alphabet_codes[i].code_) == 0) and
            (getBit(symbol_len - l - 1, alphabet_codes[i - 1].code_) == 1)) {
          node_starts.push_back({alphabet[i], l});
        }
      }
    }
    return node_starts;
  }

 private:
  std::vector<T> alphabet_;    /*!< Alphabet of the wavelet tree*/
  size_t alphabet_size_;       /*!< Size of the alphabet*/
  uint8_t alphabet_start_bit_; /*!< Bit where the alphabet starts, 0 is the
                                  least significant bit*/
  Code* d_codes_ =
      nullptr; /*!< Array of codes for each symbol in the alphabet*/
  size_t* d_counts_ = nullptr; /*!< Array of counts for each symbol*/
  uint8_t num_levels_;         /*!< Number of levels in the wavelet tree*/
  bool is_min_alphabet_; /*!< Flag to signal whether the alphabet is already
                            minimal*/
  T codes_start_;        /*!< Minimal alphabet symbol where the codes start*/
  bool is_copy_;         /*!< Flag to signal whether current object is a copy*/
  T* access_pinned_mem_pool_;
  static size_t access_mem_pool_size_;
  size_t* rank_pinned_mem_pool_;
  static size_t rank_mem_pool_size_;
  T num_nodes_until_last_level_;
  size_t* d_ranks_;
  T* d_num_nodes_at_level_ =
      nullptr; /*!< Number of nodes at each level, without counting nodes that
                  start at 0*/
  T num_ranks_;
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
template <typename T, bool UseShmemPerThread>
__global__ void fillLevelKernel(BitArray bit_array, T* const data,
                                size_t const data_size,
                                uint8_t const alphabet_start_bit,
                                uint32_t const level);

template <typename T>
__global__ void precomputeRanksKernel(WaveletTree<T> tree,
                                      NodeInfo<T>* const node_starts,
                                      size_t const total_num_nodes);

/*!
 * \brief Kernel for computing access queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param indices Indices of the symbols to be accessed.
 * \param num_indices Number of indices.
 * \param results Array to store the accessed symbols.
 */
template <typename T, bool ShmemCounts, int ThreadsPerQuery, bool ShmemOffsets>
__global__ void accessKernel(WaveletTree<T> tree, size_t* const indices,
                             size_t const num_indices, T* results,
                             size_t const alphabet_size,
                             uint32_t const num_groups,
                             uint8_t const num_levels,
                             size_t const counter_start);

/*!
 * \brief Kernel for computing rank queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param queries Array of rank queries.
 * \param num_queries Number of queries.
 * \param ranks Array to store the ranks.
 */
template <typename T, bool ShmemCounts, int ThreadsPerQuery, bool ShmemOffsets>
__global__ void rankKernel(WaveletTree<T> tree,
                           RankSelectQuery<T>* const queries,
                           size_t const num_queries, size_t* const ranks,
                           uint32_t const num_groups,
                           size_t const alphabet_size, uint8_t const num_levels,
                           size_t const counter_start);

/*!
 * \brief Kernel for computing select queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param queries Array of select queries.
 * \param num_queries Number of queries.
 * \param ranks Array to store the selected indices.
 */
template <typename T, int ThreadsPerQuery, bool ShmemRanks>
__global__ void selectKernel(WaveletTree<T> tree,
                             RankSelectQuery<T>* const queries,
                             size_t const num_queries, size_t* const results,
                             size_t const num_groups,
                             size_t const alphabet_size,
                             uint8_t const alphabet_num_bits,
                             T const codes_start, bool const is_pow_two,
                             T const num_ranks, T const num_nodes_at_start);

/****************************************************************************************
 * Implementation
 ****************************************************************************************/
typedef unsigned long long int cu_size_t;

template <typename T>
size_t WaveletTree<T>::access_mem_pool_size_ = 10'000'000;

template <typename T>
size_t WaveletTree<T>::rank_mem_pool_size_ = 10'000'000;

template <typename T>
__host__ WaveletTree<T>::WaveletTree(T* const data, size_t data_size,
                                     std::vector<T>&& alphabet,
                                     uint8_t const GPU_index)
    : alphabet_(alphabet), is_copy_(false) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type");
  assert(data_size > 0);
  assert(alphabet_.size() > 2);
  assert(std::is_sorted(alphabet_.begin(), alphabet_.end()));

  checkWarpSize(GPU_index);
  alphabet_size_ = alphabet_.size();

  bool const is_pow_two = isPowTwo(alphabet_size_);

  // Check if alphabet is already minimal
  is_min_alphabet_ =
      std::all_of(alphabet_.begin(), alphabet_.end(),
                  [i = 0](unsigned value) mutable { return value == i++; });

  std::vector<Code> codes;
  std::vector<NodeInfo<T>> node_starts;
  // make minimal alphabet
  if (not is_min_alphabet_) {
    auto min_alphabet = std::vector<T>(alphabet_size_);
    std::iota(min_alphabet.begin(), min_alphabet.end(), 0);
    codes = createMinimalCodes(min_alphabet);
    node_starts = getNodeInfos(min_alphabet, codes);
  } else {
    codes = createMinimalCodes(alphabet_);
    node_starts = getNodeInfos(alphabet_, codes);
  }

  num_levels_ = ceilLog2Host<T>(alphabet_size_);
  alphabet_start_bit_ = num_levels_ - 1;
  codes_start_ = alphabet_size_ - codes.size();

  std::vector<T> num_nodes_at_level(num_levels_ - 1);
  //  create codes and copy to device
  uint8_t* d_code_lens = nullptr;
  if (not is_pow_two) {
    gpuErrchk(cudaMalloc(&d_codes_, codes.size() * sizeof(Code)));
    gpuErrchk(cudaMemcpy(d_codes_, codes.data(), codes.size() * sizeof(Code),
                         cudaMemcpyHostToDevice));

    std::vector<uint8_t> code_lens(codes.back().code_ + 1 - codes_start_);
    for (size_t i = 0; i < alphabet_size_ - codes_start_; ++i) {
      code_lens[codes[i].code_ - codes_start_] = codes[i].len_;
    }
    gpuErrchk(cudaMalloc(&d_code_lens, code_lens.size() * sizeof(uint8_t)));
    gpuErrchk(cudaMemcpy(d_code_lens, code_lens.data(),
                         code_lens.size() * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));

    // Get number of nodes at each level
    T counter = 0;
    for (T i = 0; i < node_starts.size(); ++i) {
      if (i > 0 and node_starts[i].level_ > node_starts[i - 1].level_) {
        num_nodes_at_level[node_starts[i].level_ - 1] = counter;
        counter = 0;
      }
      counter++;
    }
    num_nodes_until_last_level_ = 0;
    for (int i = 0; i < num_nodes_at_level.size(); ++i) {
      num_nodes_until_last_level_ += num_nodes_at_level[i];
    }
    gpuErrchk(
        cudaMalloc(&d_num_nodes_at_level_, (num_levels_ - 1) * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_num_nodes_at_level_, num_nodes_at_level.data(),
                         (num_levels_ - 1) * sizeof(T),
                         cudaMemcpyHostToDevice));
  } else {
    num_nodes_until_last_level_ =
        node_starts.size() - (powTwo<T>(num_levels_ - 1) - 1);
  }
  num_ranks_ = node_starts.size();

  gpuErrchk(cudaMalloc(&d_ranks_, num_ranks_ * sizeof(size_t)));

  NodeInfo<T>* d_node_starts = nullptr;
  gpuErrchk(cudaMalloc(&d_node_starts, num_ranks_ * sizeof(NodeInfo<T>)));
  gpuErrchk(cudaMemcpy(d_node_starts, node_starts.data(),
                       num_ranks_ * sizeof(NodeInfo<T>),
                       cudaMemcpyHostToDevice));

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

  for (uint32_t l = 0; l < num_levels_; l++) {
    assert(data_size == bit_array_sizes[l]);

    if (l > 0) {
      // Perform radix sort
      cub::DeviceRadixSort::SortKeys(
          d_temp_storage, temp_storage_bytes, d_data, d_sorted_data, data_size,
          alphabet_start_bit_ + 1 - l, alphabet_start_bit_ + 1);
      //  Fill l-th bit array
      kernelCheck();
      fillLevel(bit_array, d_sorted_data, data_size, l);
    } else {
      fillLevel(bit_array, d_data, data_size, l);
    }

    if (l != (num_levels_ - 1) and
        bit_array_sizes[l] != bit_array_sizes[l + 1]) {
      //  Reduce text
      T* new_end =
          thrust::remove_if(thrust::device, d_data, d_data + data_size,
                            isNotLongEnough<T>(d_code_lens, l, codes_start_));
      data_size = static_cast<size_t>(std::distance(d_data, new_end));
    }
  }

  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_sorted_data));
  if (alphabet_size_ * sizeof(T) > temp_storage_bytes) {
    gpuErrchk(cudaFree(d_alphabet));
  }
  gpuErrchk(cudaFree(d_temp_storage));
  gpuErrchk(cudaFree(d_code_lens));

  // build rank and select structures from bit-vectors
  rank_select_ = RankSelect(std::move(bit_array), GPU_index);

  // TODO: optimize
  precomputeRanksKernel<T><<<1, 256>>>(*this, d_node_starts, num_ranks_);
  gpuErrchk(cudaFree(d_node_starts));
  kernelCheck();

  //? WHat size?
  gpuErrchk(cudaHostAlloc(&access_pinned_mem_pool_,
                          access_mem_pool_size_ * sizeof(T),
                          cudaHostAllocPortable));

  gpuErrchk(cudaHostAlloc(&rank_pinned_mem_pool_,
                          rank_mem_pool_size_ * sizeof(size_t),
                          cudaHostAllocPortable));
}

template <typename T>
__host__ WaveletTree<T>::WaveletTree(WaveletTree const& other)
    : alphabet_(other.alphabet_),
      rank_select_(other.rank_select_),
      alphabet_size_(other.alphabet_size_),
      alphabet_start_bit_(other.alphabet_start_bit_),
      num_levels_(other.num_levels_),
      d_codes_(other.d_codes_),
      d_counts_(other.d_counts_),
      d_num_nodes_at_level_(other.d_num_nodes_at_level_),
      d_ranks_(other.d_ranks_),
      is_min_alphabet_(other.is_min_alphabet_),
      codes_start_(other.codes_start_),
      access_pinned_mem_pool_(other.access_pinned_mem_pool_),
      rank_pinned_mem_pool_(other.rank_pinned_mem_pool_),
      is_copy_(true) {}

template <typename T>
WaveletTree<T>::~WaveletTree() {
  if (not is_copy_) {
    gpuErrchk(cudaFree(d_codes_));
    gpuErrchk(cudaFree(d_counts_));
    gpuErrchk(cudaFreeHost(access_pinned_mem_pool_));
    gpuErrchk(cudaFreeHost(rank_pinned_mem_pool_));
    if (d_num_nodes_at_level_ != nullptr) {
      gpuErrchk(cudaFree(d_num_nodes_at_level_));
    }
    gpuErrchk(cudaFree(d_ranks_));
  }
}

__global__ void emptyKernel() {}

__host__ [[nodiscard]] GraphContents createQueriesGraph(
    uint16_t const num_chunks, uint16_t const num_buffers) {
  assert(num_chunks > 0);
  assert(num_buffers > 0);
  assert(num_buffers <= num_chunks);

  cudaGraph_t graph;
  gpuErrchk(cudaGraphCreate(&graph, 0));

  void* placeholder_alloc;
  gpuErrchk(cudaMalloc(&placeholder_alloc, 1));

  uint8_t placeholder_host = 0;

  std::vector<cudaGraphNode_t> copy_indices_nodes(num_chunks);
  std::vector<cudaGraphNode_t> kernel_nodes(num_chunks);
  std::vector<cudaGraphNode_t> copy_results_nodes(num_chunks);

  cudaKernelNodeParams placeholder_params{.func = (void*)emptyKernel,
                                          .gridDim = dim3(1, 1, 1),
                                          .blockDim = dim3(1, 1, 1),
                                          .sharedMemBytes = 0,
                                          .kernelParams = nullptr,
                                          .extra = nullptr};

  for (uint16_t i = 0; i < num_chunks; ++i) {
    gpuErrchk(cudaGraphAddMemcpyNode1D(&copy_indices_nodes[i], graph, nullptr,
                                       0, placeholder_alloc, &placeholder_host,
                                       1, cudaMemcpyHostToDevice));

    // If there are more chunks than buffers, memcpy only allowed if no kernel
    // is using the portion of the buffer
    if (num_buffers < num_chunks and i >= num_buffers) {
      gpuErrchk(cudaGraphAddDependencies(graph, &kernel_nodes[i - num_buffers],
                                         &copy_indices_nodes[i], 1));
    }

    gpuErrchk(cudaGraphAddKernelNode(&kernel_nodes[i], graph,
                                     &copy_indices_nodes[i], 1,
                                     &placeholder_params));
    // In order to make kernel executions serial
    if (i > 0) {
      gpuErrchk(cudaGraphAddDependencies(graph, &kernel_nodes[i - 1],
                                         &kernel_nodes[i], 1));
    }

    gpuErrchk(cudaGraphAddMemcpyNode1D(
        &copy_results_nodes[i], graph, &kernel_nodes[i], 1, &placeholder_host,
        placeholder_alloc, 1, cudaMemcpyDeviceToHost));
  }

  cudaGraphExec_t graph_exec;
  gpuErrchk(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  gpuErrchk(cudaFree(placeholder_alloc));
  return GraphContents{graph_exec, copy_indices_nodes, kernel_nodes,
                       copy_results_nodes};
}

template <typename T>
template <int NumThreads>
__host__ [[nodiscard]] std::span<T> WaveletTree<T>::access(
    size_t* indices, size_t const num_indices) {
  assert(num_indices > 0);

  // CHeck if indices ptr points to pinned memory
  cudaPointerAttributes attr;
  gpuErrchk(cudaPointerGetAttributes(&attr, indices));
  bool const is_pinned_mem = attr.type == cudaMemoryTypeHost;

  std::thread cpu_mem_thread([&]() {
    if (not is_pinned_mem) {
      gpuErrchk(cudaHostRegister(indices, num_indices * sizeof(size_t),
                                 cudaHostRegisterPortable));
    }
  });

  // should rarely happen
  if (num_indices > access_mem_pool_size_) {
    gpuErrchk(cudaFreeHost(access_pinned_mem_pool_));
    access_mem_pool_size_ = num_indices;
    gpuErrchk(cudaHostAlloc(&access_pinned_mem_pool_,
                            access_mem_pool_size_ * sizeof(T),
                            cudaHostAllocPortable));
  }

  // TODO: find good heuristic
  //  Divide indices into chunks
  uint32_t const num_chunks = num_indices < 10 ? 1 : 10;
  uint8_t const num_buffers = std::min(num_chunks, 2U);
  size_t const chunk_size = num_indices / num_chunks;
  size_t const last_chunk_size = chunk_size + num_indices % num_chunks;

  size_t* d_indices;
  T* d_results;

  std::thread gpu_alloc_thread([&]() {
    gpuErrchk(
        cudaMalloc(&d_indices, last_chunk_size * num_buffers * sizeof(size_t)));

    gpuErrchk(cudaMalloc(&d_results, num_indices * sizeof(T)));
    size_t const tmp = 0;
    gpuErrchk(cudaMemcpyToSymbol(access_counter, &tmp, sizeof(size_t)));
  });

  T* results = access_pinned_mem_pool_;

  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                  accessKernel<T, true, NumThreads, true>));

  auto maxThreadsPerBlockAccess =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  maxThreadsPerBlockAccess =
      findLargestDivisor(kMaxTPB, maxThreadsPerBlockAccess);

  size_t const counts_size = sizeof(size_t) * alphabet_size_;

  size_t const offsets_size = sizeof(size_t) * num_levels_;

  struct cudaDeviceProp prop = getDeviceProperties();
  gpuErrchk(cudaFuncSetAttribute(
      accessKernel<T, true, NumThreads, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      prop.sharedMemPerBlockOptin - funcAttrib.sharedSizeBytes));
  gpuErrchk(cudaFuncSetAttribute(
      accessKernel<T, false, NumThreads, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      prop.sharedMemPerBlockOptin - funcAttrib.sharedSizeBytes));

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;

  size_t shmem_per_block = funcAttrib.sharedSizeBytes;

  bool const offsets_shmem =
      (offsets_size + shmem_per_block) * kMinBPM <= max_shmem_per_SM;

  auto min_block_size = kMinTPB;
  if (offsets_shmem) {
    shmem_per_block += offsets_size;
    for (uint32_t block_size = kMaxTPB; block_size >= kMinTPB;
         block_size /= 2) {
      auto const blocks_per_sm = kMinBPM * kMaxTPB / block_size;
      if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
        min_block_size = 2 * block_size;
        break;
      }
    }
  }

  bool const counts_shmem =
      (counts_size + shmem_per_block) * kMinBPM <= max_shmem_per_SM;

  if (counts_shmem) {
    shmem_per_block += counts_size;
    for (uint32_t block_size = kMaxTPB; block_size >= kMinTPB;
         block_size /= 2) {
      auto const blocks_per_sm = kMinBPM * kMaxTPB / block_size;
      if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
        min_block_size = 2 * block_size;
        break;
      }
    }
  }

  int num_blocks, threads_per_block;
  IdealConfigs ideal_configs = getIdealConfigs(prop.name);
  if (ideal_configs.ideal_TPB_accessKernel != 0) {
    size_t const num_warps =
        std::min(ideal_configs.ideal_tot_threads_accessKernel / WS,
                 static_cast<size_t>((chunk_size * NumThreads) / WS));
    if (ideal_configs.ideal_TPB_accessKernel < min_block_size) {
      std::tie(num_blocks, threads_per_block) =
          getLaunchConfig(num_warps, min_block_size, maxThreadsPerBlockAccess);
    } else {
      threads_per_block = ideal_configs.ideal_TPB_accessKernel;
      num_blocks = (num_warps * WS) / threads_per_block;
    }
  } else {
    // Make the minimum block size a multiple of WS
    std::tie(num_blocks, threads_per_block) = getLaunchConfig(
        std::min((chunk_size * NumThreads + WS - 1) / WS,
                 static_cast<size_t>((prop.maxThreadsPerMultiProcessor *
                                      prop.multiProcessorCount) /
                                     WS)),
        min_block_size, maxThreadsPerBlockAccess);
  }

  uint32_t const num_groups = (num_blocks * threads_per_block) / NumThreads;

  GraphContents graph_contents;
  if (queries_graph_cache.find(num_chunks) == queries_graph_cache.end()) {
    graph_contents = createQueriesGraph(num_chunks, num_buffers);
    queries_graph_cache[num_chunks] = graph_contents;
  } else {
    graph_contents = queries_graph_cache[num_chunks];
  }

  // Change parameters of graph
  void* kernel_params_base[8] = {
      this,         nullptr, 0, nullptr, &alphabet_size_, (void*)&num_groups,
      &num_levels_, 0};

  auto kernel_node_params_base = cudaKernelNodeParams{
      .func = nullptr,
      .gridDim = dim3(num_blocks, 1, 1),
      .blockDim = dim3(threads_per_block, 1, 1),
      .sharedMemBytes =
          static_cast<uint32_t>(shmem_per_block - funcAttrib.sharedSizeBytes),
      .kernelParams = nullptr,
      .extra = nullptr};

  if (offsets_shmem) {
    if (counts_shmem) {
      kernel_node_params_base.func =
          (void*)(&accessKernel<T, true, NumThreads, true>);
    } else {
      kernel_node_params_base.func =
          (void*)(&accessKernel<T, false, NumThreads, true>);
    }
  } else {
    kernel_node_params_base.func =
        (void*)(&accessKernel<T, false, NumThreads, false>);
  }

  // Allocations must be done before setting parameters
  gpu_alloc_thread.join();
  cpu_mem_thread.join();
  for (uint16_t i = 0; i < num_chunks; ++i) {
    auto const current_chunk_size =
        i == num_chunks - 1 ? last_chunk_size : chunk_size;
    size_t* current_d_indices = d_indices + last_chunk_size * (i % num_buffers);

    gpuErrchk(cudaGraphExecMemcpyNodeSetParams1D(
        graph_contents.graph_exec, graph_contents.copy_indices_nodes[i],
        current_d_indices, indices + i * chunk_size,
        current_chunk_size * sizeof(size_t), cudaMemcpyHostToDevice));

    auto kernel_node_params = kernel_node_params_base;
    T* current_d_results = d_results + i * chunk_size;

    size_t current_d_indices_offset = i * chunk_size;

    void* kernel_params[8] = {kernel_params_base[0],
                              &current_d_indices,
                              current_chunk_size == last_chunk_size
                                  ? (size_t*)&last_chunk_size
                                  : (size_t*)&chunk_size,
                              &current_d_results,
                              kernel_params_base[4],
                              kernel_params_base[5],
                              kernel_params_base[6],
                              &current_d_indices_offset};

    kernel_node_params.kernelParams = kernel_params;

    gpuErrchk(cudaGraphExecKernelNodeSetParams(graph_contents.graph_exec,
                                               graph_contents.kernel_nodes[i],
                                               &kernel_node_params));

    gpuErrchk(cudaGraphExecMemcpyNodeSetParams1D(
        graph_contents.graph_exec, graph_contents.copy_results_nodes[i],
        results + i * chunk_size, current_d_results,
        current_chunk_size * sizeof(T), cudaMemcpyDeviceToHost));
  }
  gpuErrchk(cudaGraphLaunch(graph_contents.graph_exec, 0));
  kernelCheck();

  std::thread end_thread([&]() {
    gpuErrchk(cudaFree(d_indices));
    gpuErrchk(cudaFree(d_results));
    if (not is_pinned_mem) {
      gpuErrchk(cudaHostUnregister(indices));
    }
  });

  if (not is_min_alphabet_) {
#pragma omp parallel for
    for (size_t i = 0; i < num_indices; ++i) {
      results[i] = alphabet_[results[i]];
    }
  }
  end_thread.join();
  return std::span<T>(results, num_indices);
}

template <typename T>
template <int NumThreads>
__host__ [[nodiscard]] std::span<size_t> WaveletTree<T>::rank(
    RankSelectQuery<T>* queries, size_t const num_queries) {
  assert(std::all_of(queries, queries + num_queries,
                     [&](const RankSelectQuery<T>& s) {
                       return s.index_ < rank_select_.bit_array_.sizeHost(0);
                     }));

  cudaPointerAttributes attr;
  gpuErrchk(cudaPointerGetAttributes(&attr, queries));
  bool const is_pinned_mem = attr.type == cudaMemoryTypeHost;

  std::thread cpu_mem_thread([&]() {
    if (not is_pinned_mem) {
      gpuErrchk(cudaHostRegister(queries,
                                 num_queries * sizeof(RankSelectQuery<T>),
                                 cudaHostRegisterPortable));
    }
  });

  // should rarely happen
  if (num_queries > rank_mem_pool_size_) {
    gpuErrchk(cudaFreeHost(rank_pinned_mem_pool_));
    rank_mem_pool_size_ = num_queries;
    gpuErrchk(cudaHostAlloc(&rank_pinned_mem_pool_,
                            rank_mem_pool_size_ * sizeof(size_t),
                            cudaHostAllocPortable));
  }

  // TODO: find good heuristic
  //  Divide indices into chunks
  uint32_t const num_chunks = num_queries < 10 ? 1 : 10;
  uint8_t const num_buffers = std::min(num_chunks, 2U);
  size_t const chunk_size = num_queries / num_chunks;
  size_t const last_chunk_size = chunk_size + num_queries % num_chunks;

  RankSelectQuery<T>* d_queries;
  size_t* d_results;

  std::thread gpu_alloc_thread([&]() {
    gpuErrchk(cudaMalloc(&d_queries, last_chunk_size * num_buffers *
                                         sizeof(RankSelectQuery<T>)));

    gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(size_t)));
    size_t const tmp = 0;
    gpuErrchk(cudaMemcpyToSymbol(rank_counter, &tmp, sizeof(size_t)));
  });

  size_t* results = rank_pinned_mem_pool_;

  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                  rankKernel<T, true, NumThreads, true>));
  uint32_t maxThreadsPerBlock =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  maxThreadsPerBlock = findLargestDivisor(kMaxTPB, maxThreadsPerBlock);

  size_t const counts_size = sizeof(size_t) * alphabet_size_;

  size_t const offsets_size = sizeof(size_t) * num_levels_;

  struct cudaDeviceProp prop = getDeviceProperties();
  gpuErrchk(cudaFuncSetAttribute(
      rankKernel<T, true, NumThreads, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      prop.sharedMemPerBlockOptin - funcAttrib.sharedSizeBytes));
  gpuErrchk(cudaFuncSetAttribute(
      rankKernel<T, false, NumThreads, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      prop.sharedMemPerBlockOptin - funcAttrib.sharedSizeBytes));

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;

  size_t shmem_per_block = funcAttrib.sharedSizeBytes;

  bool const offsets_shmem =
      (offsets_size + shmem_per_block) * kMinBPM <= max_shmem_per_SM;

  auto min_block_size = kMinTPB;
  if (offsets_shmem) {
    shmem_per_block += offsets_size;
    for (uint32_t block_size = kMaxTPB; block_size >= kMinTPB;
         block_size /= 2) {
      auto const blocks_per_sm = kMinBPM * kMaxTPB / block_size;
      if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
        min_block_size = 2 * block_size;
        break;
      }
    }
  }

  bool const counts_shmem =
      (counts_size + shmem_per_block) * kMinBPM <= max_shmem_per_SM;

  if (counts_shmem) {
    shmem_per_block += counts_size;
    for (uint32_t block_size = kMaxTPB; block_size >= kMinTPB;
         block_size /= 2) {
      auto const blocks_per_sm = kMinBPM * kMaxTPB / block_size;
      if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
        min_block_size = 2 * block_size;
        break;
      }
    }
  }

  int num_blocks, threads_per_block;
  IdealConfigs ideal_configs = getIdealConfigs(prop.name);
  if (ideal_configs.ideal_TPB_rankKernel != 0) {
    size_t const num_warps =
        std::min(ideal_configs.ideal_tot_threads_rankKernel / WS,
                 static_cast<size_t>((chunk_size * NumThreads) / WS));
    if (ideal_configs.ideal_TPB_rankKernel < min_block_size) {
      std::tie(num_blocks, threads_per_block) =
          getLaunchConfig(num_warps, min_block_size, maxThreadsPerBlock);
    } else {
      threads_per_block = ideal_configs.ideal_TPB_rankKernel;
      num_blocks = (num_warps * WS) / threads_per_block;
    }
  } else {
    // Make the minimum block size a multiple of WS
    std::tie(num_blocks, threads_per_block) = getLaunchConfig(
        std::min((chunk_size * NumThreads + WS - 1) / WS,
                 static_cast<size_t>((prop.maxThreadsPerMultiProcessor *
                                      prop.multiProcessorCount) /
                                     WS)),
        min_block_size, maxThreadsPerBlock);
  }

  uint32_t const num_groups = (num_blocks * threads_per_block) / NumThreads;

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

  GraphContents graph_contents;
  if (queries_graph_cache.find(num_chunks) == queries_graph_cache.end()) {
    graph_contents = createQueriesGraph(num_chunks, num_buffers);
    queries_graph_cache[num_chunks] = graph_contents;
  } else {
    graph_contents = queries_graph_cache[num_chunks];
  }

  // Change parameters of graph
  void* kernel_params_base[8] = {
      this,         nullptr, 0, nullptr, (void*)&num_groups, &alphabet_size_,
      &num_levels_, 0};

  auto kernel_node_params_base = cudaKernelNodeParams{
      .func = nullptr,
      .gridDim = dim3(num_blocks, 1, 1),
      .blockDim = dim3(threads_per_block, 1, 1),
      .sharedMemBytes =
          static_cast<uint32_t>(shmem_per_block - funcAttrib.sharedSizeBytes),
      .kernelParams = nullptr,
      .extra = nullptr};

  if (offsets_shmem) {
    if (counts_shmem) {
      kernel_node_params_base.func =
          (void*)(&rankKernel<T, true, NumThreads, true>);
    } else {
      kernel_node_params_base.func =
          (void*)(&rankKernel<T, false, NumThreads, true>);
    }
  } else {
    kernel_node_params_base.func =
        (void*)(&rankKernel<T, false, NumThreads, false>);
  }

  // Allocations must be done before setting parameters
  gpu_alloc_thread.join();
  cpu_mem_thread.join();
  for (uint16_t i = 0; i < num_chunks; ++i) {
    auto const current_chunk_size =
        i == num_chunks - 1 ? last_chunk_size : chunk_size;
    RankSelectQuery<T>* current_d_queries =
        d_queries + last_chunk_size * (i % num_buffers);

    gpuErrchk(cudaGraphExecMemcpyNodeSetParams1D(
        graph_contents.graph_exec, graph_contents.copy_indices_nodes[i],
        current_d_queries, queries + i * chunk_size,
        current_chunk_size * sizeof(RankSelectQuery<T>),
        cudaMemcpyHostToDevice));

    auto kernel_node_params = kernel_node_params_base;
    size_t* current_d_results = d_results + i * chunk_size;

    size_t current_d_queries_offset = i * chunk_size;

    void* kernel_params[8] = {kernel_params_base[0],
                              &current_d_queries,
                              current_chunk_size == last_chunk_size
                                  ? (size_t*)&last_chunk_size
                                  : (size_t*)&chunk_size,
                              &current_d_results,
                              kernel_params_base[4],
                              kernel_params_base[5],
                              kernel_params_base[6],
                              &current_d_queries_offset};

    kernel_node_params.kernelParams = kernel_params;

    gpuErrchk(cudaGraphExecKernelNodeSetParams(graph_contents.graph_exec,
                                               graph_contents.kernel_nodes[i],
                                               &kernel_node_params));

    gpuErrchk(cudaGraphExecMemcpyNodeSetParams1D(
        graph_contents.graph_exec, graph_contents.copy_results_nodes[i],
        results + i * chunk_size, current_d_results,
        current_chunk_size * sizeof(size_t), cudaMemcpyDeviceToHost));
  }
  gpuErrchk(cudaGraphLaunch(graph_contents.graph_exec, 0));
  kernelCheck();

  gpuErrchk(cudaFree(d_queries));
  gpuErrchk(cudaFree(d_results));
  if (not is_pinned_mem) {
    gpuErrchk(cudaHostUnregister(queries));
  }

  return std::span<size_t>(results, num_queries);
}

// TODO: when documenting, emphasize the power of sorting queries
template <typename T>
template <int ThreadsPerQuery>
__host__ [[nodiscard]] std::vector<size_t> WaveletTree<T>::select(
    std::vector<RankSelectQuery<T>>& queries) {
  assert(std::all_of(queries.begin(), queries.end(),
                     [](const RankSelectQuery<T>& s) { return s.index_ > 0; }));
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                  selectKernel<T, ThreadsPerQuery, true>));
  uint32_t maxThreadsPerBlock =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  maxThreadsPerBlock = findLargestDivisor(kMaxTPB, maxThreadsPerBlock);
  // launch kernel with 1 warp per index
  size_t const num_queries = queries.size();
  auto prop = getDeviceProperties();

  gpuErrchk(cudaFuncSetAttribute(
      selectKernel<T, ThreadsPerQuery, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      prop.sharedMemPerBlockOptin - funcAttrib.sharedSizeBytes));

  size_t const ranks_size = sizeof(size_t) * num_ranks_;

  size_t const total_shmem_per_block = ranks_size + funcAttrib.sharedSizeBytes;

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;

  bool const ranks_shmem = total_shmem_per_block * kMinBPM <= max_shmem_per_SM;

  auto min_block_size = kMinTPB;
  if (ranks_shmem) {
    for (uint32_t block_size = kMaxTPB; block_size >= kMinTPB;
         block_size /= 2) {
      auto const blocks_per_sm = kMinBPM * kMaxTPB / block_size;
      if (total_shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
        min_block_size = 2 * block_size;
        break;
      }
    }
  }

  auto [num_blocks, threads_per_block] = getLaunchConfig(
      (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount + WS - 1) /
          WS,
      min_block_size, maxThreadsPerBlock);

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

  if (ranks_shmem) {
    selectKernel<T, ThreadsPerQuery, true>
        <<<num_blocks, threads_per_block, ranks_size>>>(
            *this, d_queries, num_queries, d_results,
            num_blocks * threads_per_block / ThreadsPerQuery, alphabet_size_,
            num_levels_, codes_start_, isPowTwo<size_t>(alphabet_size_),
            num_ranks_, num_nodes_until_last_level_);
  } else {
    selectKernel<T, ThreadsPerQuery, false><<<num_blocks, threads_per_block>>>(
        *this, d_queries, num_queries, d_results,
        num_blocks * threads_per_block / ThreadsPerQuery, alphabet_size_,
        num_levels_, codes_start_, isPowTwo<size_t>(alphabet_size_), num_ranks_,
        num_nodes_until_last_level_);
  }
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
__device__ uint8_t WaveletTree<T>::getNumLevels() const {
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
  struct cudaDeviceProp prop = getDeviceProperties();

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
  auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
  size_t const hist_size = sizeof(size_t) * alphabet_size_;

  auto ideal_configs = getIdealConfigs(prop.name);

  int num_blocks, threads_per_block;
  if (ideal_configs.ideal_tot_threads_computeGlobalHistogramKernel > 0 and
      ideal_configs.ideal_TPB_computeGlobalHistogramKernel > 0) {
    std::tie(num_blocks, threads_per_block) = getLaunchConfig(
        std::min(
            (data_size + WS - 1) / WS,
            ideal_configs.ideal_tot_threads_computeGlobalHistogramKernel / WS),
        ideal_configs.ideal_TPB_computeGlobalHistogramKernel,
        ideal_configs.ideal_TPB_computeGlobalHistogramKernel);
  } else {
    auto maxThreadsPerBlockHist =
        std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
    maxThreadsPerBlockHist =
        findLargestDivisor(kMaxTPB, maxThreadsPerBlockHist);

    auto const hists_per_SM = max_shmem_per_SM / hist_size;

    auto min_block_size =
        hists_per_SM < kMinBPM
            ? kMinTPB
            : std::max(kMinTPB, static_cast<uint32_t>(max_threads_per_SM /
                                                      hists_per_SM));

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

    std::tie(num_blocks, threads_per_block) =
        getLaunchConfig(num_warps, min_block_size, maxThreadsPerBlockHist);
  }

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

template <typename T>
__host__ void WaveletTree<T>::fillLevel(BitArray bit_array, T* const data,
                                        size_t const data_size,
                                        uint32_t const level) {
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, fillLevelKernel<T, true>));
  uint32_t maxThreadsPerBlockFillLevel =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, fillLevelKernel<T, false>));
  maxThreadsPerBlockFillLevel =
      std::min(maxThreadsPerBlockFillLevel,
               static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  maxThreadsPerBlockFillLevel =
      findLargestDivisor(kMaxTPB, maxThreadsPerBlockFillLevel);

  struct cudaDeviceProp prop = getDeviceProperties();
  if (level == 0) {
    gpuErrchk(cudaFuncSetAttribute(
        fillLevelKernel<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        prop.sharedMemPerBlockOptin - funcAttrib.sharedSizeBytes));
  }

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
  auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
  auto shmem_per_thread = sizeof(uint32_t) * 8 * sizeof(T);

  // Pad shmem banks if memory suffices
  size_t padded_shmem = std::numeric_limits<size_t>::max();
  if constexpr (kBankSizeBytes <= sizeof(T)) {
    // If the bank size is smaller than or equal to T, one element of padding
    // per thread is needed.
    padded_shmem =
        shmem_per_thread * max_threads_per_SM + sizeof(T) * max_threads_per_SM;
  } else {
    padded_shmem = shmem_per_thread * max_threads_per_SM +
                   shmem_per_thread * max_threads_per_SM / kBanksPerLine;
  }
  bool const enough_shmem =
      (padded_shmem + funcAttrib.sharedSizeBytes) <= max_shmem_per_SM;
  if (enough_shmem) {
    shmem_per_thread = padded_shmem / max_threads_per_SM;
  }

  int num_blocks, threads_per_block;

  auto ideal_configs = getIdealConfigs(prop.name);

  if (ideal_configs.ideal_TPB_fillLevelKernel > 0 and
      ideal_configs.ideal_tot_threads_fillLevelKernel > 0) {
    size_t const num_warps =
        std::min((data_size + WS - 1) / WS,
                 ideal_configs.ideal_tot_threads_fillLevelKernel / WS);

    std::tie(num_blocks, threads_per_block) =
        getLaunchConfig(num_warps, ideal_configs.ideal_TPB_fillLevelKernel,
                        ideal_configs.ideal_TPB_fillLevelKernel);

  } else {
    size_t const num_warps = std::min(
        (data_size + WS - 1) / WS,
        static_cast<size_t>(
            (max_threads_per_SM * prop.multiProcessorCount + WS - 1) / WS));

    std::tie(num_blocks, threads_per_block) =
        getLaunchConfig(num_warps, kMinTPB, maxThreadsPerBlockFillLevel);
  }

  if (enough_shmem) {
    fillLevelKernel<T, true><<<num_blocks, threads_per_block,
                               shmem_per_thread * threads_per_block>>>(
        bit_array, data, data_size, alphabet_start_bit_, level);
  } else {
    fillLevelKernel<T, false><<<num_blocks, threads_per_block,
                                sizeof(uint32_t) * (threads_per_block / WS)>>>(
        bit_array, data, data_size, alphabet_start_bit_, level);
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
    if constexpr (UseShmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ <= 520
      atomicAdd((cu_size_t*)&shared_hist[offset + char_data], size_t(1));
#else
      atomicAdd_block((cu_size_t*)&shared_hist[offset + char_data], size_t(1));
#endif
    } else {
      atomicAdd((cu_size_t*)&counts[char_data], size_t(1));
    }
    if constexpr (not isPowTwo) {
      if (char_data >= tree.getCodesStart()) {
        char_data = tree.encode(char_data).code_;
      }
    }
    if constexpr (not isMinAlphabet or not isPowTwo) {
      data[i] = char_data;
    }
  }

  if constexpr (UseShmem) {
    __syncthreads();
    // Reduce shared histograms to first one
    for (size_t i = threadIdx.x; i < alphabet_size; i += blockDim.x) {
      size_t sum = shared_hist[i];
      for (size_t j = 1; j < hists_per_block; ++j) {
        sum += shared_hist[j * alphabet_size + i];
      }
      atomicAdd((cu_size_t*)&counts[i], sum);
    }
  }
}

// TODO reduce register usage
template <typename T, bool UseShmemPerThread>
__global__ LB(MAX_TPB,
              MIN_BPM) void fillLevelKernel(BitArray bit_array, T* const data,
                                            size_t const data_size,
                                            uint8_t const alphabet_start_bit,
                                            uint32_t const level) {
  assert(blockDim.x % WS == 0);
  size_t const offset = bit_array.getOffset(level);

  if constexpr (UseShmemPerThread) {
    extern __shared__ __align__(sizeof(T)) uint8_t my_smem[];
    T* data_slice = reinterpret_cast<T*>(my_smem);
    size_t const slice_size = blockDim.x * sizeof(uint32_t) * 8;
    for (size_t i = blockIdx.x * slice_size; i < data_size;
         i += gridDim.x * slice_size) {
      // Load text slice to shared memory
      uint32_t elems_to_skip = 0;
      for (size_t j = threadIdx.x; j < min(data_size - i, slice_size);
           j += blockDim.x) {
        if constexpr (kBankSizeBytes < sizeof(T)) {
          elems_to_skip = j / kBanksPerLine;
        } else {
          elems_to_skip = j / ((kBankSizeBytes / sizeof(T)) * kBanksPerLine);
        }
        data_slice[j + elems_to_skip] = data[j + i];
      }
      __syncthreads();
      uint32_t word = 0;
      uint32_t const start = threadIdx.x * sizeof(uint32_t) * 8;
      uint32_t const end = min(start + sizeof(uint32_t) * 8, data_size - i);
      elems_to_skip = 0;
      if constexpr (kBankSizeBytes < sizeof(T)) {
        elems_to_skip = start / kBanksPerLine;
      } else {
        elems_to_skip = start / ((kBankSizeBytes / sizeof(T)) * kBanksPerLine);
      }
      // Start from the end, since LSB is the first bit
      if (end > start) {
        for (uint32_t j = end - 1; j > start; --j) {
          word |= static_cast<uint8_t>(getBit(alphabet_start_bit - level,
                                              data_slice[j + elems_to_skip]));
          word <<= 1;
        }
        word |= static_cast<uint8_t>(getBit(alphabet_start_bit - level,
                                            data_slice[start + elems_to_skip]));
        bit_array.writeWordAtBit(level, i + start, word, offset);
      }
      __syncthreads();
    }
  } else {
    extern __shared__ uint32_t shared_words[];
    // Round data size to multiple of block size
    size_t const data_size_rounded =
        ((data_size + (blockDim.x - 1)) / blockDim.x) * blockDim.x;
    size_t const global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const num_threads = gridDim.x * blockDim.x;
    uint8_t const local_t_id = threadIdx.x % WS;
    uint32_t const warps_per_block = blockDim.x / WS;
    size_t const bit_array_size = bit_array.size(level);
    // Each warp processes a block of data
    for (size_t i = global_t_id; i < data_size_rounded; i += num_threads) {
      T code = 0;
      if (i < data_size) {
        code = data[i];
      }

      // Warp vote to all the bits that need to get written to a word
      uint32_t word =
          __ballot_sync(~0, getBit(alphabet_start_bit - level, code));

      if (local_t_id == 0) {
        shared_words[threadIdx.x / WS] = word;
      }
      __syncthreads();
      if (threadIdx.x < warps_per_block) {
        size_t const index = (i - threadIdx.x) + WS * threadIdx.x;
        if (index < bit_array_size) {
          bit_array.writeWordAtBit(level, index, shared_words[threadIdx.x],
                                   offset);
        }
      }
      __syncthreads();
    }
  }
}

template <typename T>
__global__ LB(MAX_TPB, MIN_BPM) void precomputeRanksKernel(
    WaveletTree<T> tree, NodeInfo<T>* const node_starts,
    size_t const total_num_nodes) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type");
  assert(blockDim.x % WS == 0);

  uint32_t const global_t_id = (blockIdx.x * blockDim.x + threadIdx.x);
  uint32_t const num_threads = gridDim.x * blockDim.x;

  for (size_t i = global_t_id; i < total_num_nodes; i += num_threads) {
    auto const node_info = node_starts[i];
    size_t const char_counts = tree.getCounts(node_info.start_);
    uint8_t const level = node_info.level_;
    size_t const offset = tree.rank_select_.bit_array_.getOffset(level);

#pragma nv_diag_suppress 174
    tree.setPrecomputedRank(i, tree.rank_select_.template rank<1, false, 0>(
                                   level, char_counts, offset));
#pragma nv_diag_default 174
  }
}

template <typename T, bool ShmemCounts, int ThreadsPerQuery, bool ShmemOffsets>
__global__ LB(MAX_TPB, MIN_BPM) void accessKernel(
    WaveletTree<T> tree, size_t* const indices, size_t const num_indices,
    T* results, size_t const alphabet_size, uint32_t const num_groups,
    uint8_t const num_levels, size_t const counter_start) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type");
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t shmem[];

  if constexpr (ShmemOffsets) {
    for (uint32_t i = threadIdx.x; i < num_levels; i += blockDim.x) {
      shmem[i] = tree.rank_select_.bit_array_.getOffset(i);
    }

    if constexpr (ShmemCounts) {
      for (size_t i = threadIdx.x; i < alphabet_size; i += blockDim.x) {
        shmem[num_levels + i] = tree.getCounts(i);
      }
    }
    __syncthreads();
  }

  // TODO: If only 1 TPQ, do atomic add inside of while statement.
  uint8_t const local_t_id = threadIdx.x % ThreadsPerQuery;
  while (access_counter < num_indices + counter_start) {
    size_t i;
    if (local_t_id == 0) {
      i = atomicAdd((cu_size_t*)&access_counter, size_t(1)) - counter_start;
    }
    if constexpr (ThreadsPerQuery > 1) {
      uint32_t mask = ~0;
      if constexpr (ThreadsPerQuery < WS) {
        mask = ((1 << ThreadsPerQuery) - 1)
               << (ThreadsPerQuery * ((threadIdx.x % WS) / ThreadsPerQuery));
      }
      shareVar<size_t>(local_t_id == 0, i, mask);
    }
    if (i >= num_indices) {
      break;
    }
    size_t index = indices[i];

    T char_start = 0;
    T char_end = alphabet_size - 1;
    for (uint8_t l = 0; l < num_levels; ++l) {
      size_t char_counts;
      if constexpr (ShmemCounts) {
        char_counts = shmem[num_levels + char_start];
      } else {
        char_counts = tree.getCounts(char_start);
      }
      size_t offset;
      if constexpr (ShmemOffsets) {
        offset = shmem[l];
      } else {
        offset = tree.rank_select_.bit_array_.getOffset(l);
      }
      if (char_end - char_start < 2) {
        if (char_end - char_start > 0 and
            tree.rank_select_.bit_array_.access(l, char_counts + index,
                                                offset) == true) {
          char_start++;
        }
        break;
      }
#pragma nv_diag_suppress 174
      size_t const start =
          tree.rank_select_.template rank<ThreadsPerQuery, false, 0>(
              l, char_counts, offset);
      RankResult const result =
          tree.rank_select_.template rank<ThreadsPerQuery, true, 0>(
              l, char_counts + index, offset);
#pragma nv_diag_default 174
      size_t const pos = result.rank;
      bool const bit_at_index = result.bit;
      T const diff = getPrevPowTwo<T>(char_end - char_start + 1);
      if (bit_at_index == false) {
        index = pos - start;
        char_end = char_start + (diff - 1);
      } else {
        index -= pos - start;
        char_start += diff;
      }
    }
    if (local_t_id == 0) {
      results[i] = char_start;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin((cu_size_t*)&access_counter, num_indices + counter_start);
  }
}

template <typename T, bool ShmemCounts, int ThreadsPerQuery, bool ShmemOffsets>
__global__ LB(MAX_TPB, MIN_BPM) void rankKernel(
    WaveletTree<T> tree, RankSelectQuery<T>* const queries,
    size_t const num_queries, size_t* const ranks, uint32_t const num_groups,
    size_t const alphabet_size, uint8_t const num_levels,
    size_t const counter_start) {
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t shmem[];

  if constexpr (ShmemOffsets) {
    for (uint32_t i = threadIdx.x; i < num_levels; i += blockDim.x) {
      shmem[i] = tree.rank_select_.bit_array_.getOffset(i);
    }

    if constexpr (ShmemCounts) {
      for (size_t i = threadIdx.x; i < alphabet_size; i += blockDim.x) {
        shmem[num_levels + i] = tree.getCounts(i);
      }
    }
    __syncthreads();
  }

  uint8_t const local_t_id = threadIdx.x % ThreadsPerQuery;

  while (rank_counter < num_queries + counter_start) {
    size_t i;
    if (local_t_id == 0) {
      i = atomicAdd((cu_size_t*)&rank_counter, size_t(1)) - counter_start;
    }
    if constexpr (ThreadsPerQuery > 1) {
      uint32_t mask = ~0;
      if constexpr (ThreadsPerQuery < WS) {
        mask = ((1 << ThreadsPerQuery) - 1)
               << (ThreadsPerQuery * ((threadIdx.x % WS) / ThreadsPerQuery));
      }
      shareVar<size_t>(local_t_id == 0, i, mask);
    }
    if (i >= num_queries) {
      break;
    }
    RankSelectQuery<T> query = queries[i];

    T char_start = 0;
    T char_end = alphabet_size - 1;
    T char_split;
    size_t start, pos;
    for (uint8_t l = 0; l < num_levels; ++l) {
      if (char_end - char_start == 0) {
        break;
      }
      size_t char_counts;
      if constexpr (ShmemCounts) {
        char_counts = shmem[num_levels + char_start];
      } else {
        char_counts = tree.getCounts(char_start);
      }
      size_t offset;
      if constexpr (ShmemOffsets) {
        offset = shmem[l];
      } else {
        offset = tree.rank_select_.bit_array_.getOffset(l);
      }
#pragma nv_diag_suppress 174
      start = tree.rank_select_.template rank<ThreadsPerQuery, false, 0>(
          l, char_counts, offset);
      pos = tree.rank_select_.template rank<ThreadsPerQuery, false, 0>(
          l, char_counts + query.index_, offset);
#pragma nv_diag_default 174
      char_split = char_start + getPrevPowTwo<T>(char_end - char_start + 1);
      if (query.symbol_ < char_split) {
        query.index_ = pos - start;
        char_end = char_split - 1;
      } else {
        query.index_ -= pos - start;
        char_start = char_split;
      }
    }
    if (local_t_id == 0) {
      ranks[i] = query.index_;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin((cu_size_t*)&rank_counter, num_queries + counter_start);
  }
}

template <typename T, bool IsPowTwo>
__device__ T getPrevCharStart(T const char_start, bool const is_rightmost_child,
                              size_t const alphabet_size, uint8_t const level,
                              uint8_t const num_levels,
                              uint8_t const code_len) {
  if constexpr (IsPowTwo) {
    T const node_lens = 1ULL << (num_levels - level);
    // Round char_start down to the nearest node start
    return char_start & ~(node_lens - 1);
  } else {
    if (is_rightmost_child) {
      uint32_t new_start = 0;
      for (uint32_t l = 0; l < level; l++) {
        new_start += getPrevPowTwo(alphabet_size - new_start);
      }
      return new_start;
    } else {
      return char_start - powTwo<uint32_t>(code_len - 1 - level);
    }
  }
}

// TODO: add counts to shmem
template <typename T, int ThreadsPerQuery, bool ShmemRanks>
__global__ LB(MAX_TPB, MIN_BPM) void selectKernel(
    WaveletTree<T> tree, RankSelectQuery<T>* const queries,
    size_t const num_queries, size_t* const results, size_t const num_groups,
    size_t const alphabet_size, uint8_t const alphabet_num_bits,
    T const codes_start, bool const is_pow_two, T const num_ranks,
    T const num_nodes_at_start) {
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t shmem[];
  if constexpr (ShmemRanks) {
    for (uint32_t i = threadIdx.x; i < num_ranks; i += blockDim.x) {
      shmem[i] = tree.getPrecomputedRank(i);
    }
    __syncthreads();
  }

  size_t const global_group_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerQuery;
  uint8_t const local_t_id = threadIdx.x % ThreadsPerQuery;

  for (size_t i = global_group_id; i < num_queries; i += num_groups) {
    RankSelectQuery<T> query = queries[i];
    if (local_t_id == 0 and
        query.index_ > tree.getTotalAppearances(query.symbol_)) {
      results[i] = tree.rank_select_.bit_array_.size(0);
      continue;
    }
    typename WaveletTree<T>::Code code{alphabet_num_bits, query.symbol_};
    if (query.symbol_ >= codes_start) {
      code = tree.encode(query.symbol_);
    }

    T char_start = query.symbol_;
    size_t start;
    T ranks_start = num_nodes_at_start;
    int16_t l = code.len_ - 1;
    for (int16_t level = alphabet_num_bits - 2; level >= l; --level) {
      ranks_start -=
          is_pow_two ? powTwo<T>(level) - 1 : tree.getNumNodesAtLevel(level);
    }
    for (; l >= 0; --l) {
      size_t const offset = tree.rank_select_.bit_array_.getOffset(l);
      // If it's a right child
      if (getBit<T>(alphabet_num_bits - 1 - l, code.code_) == true) {
        if (is_pow_two) {
          char_start = getPrevCharStart<T, true>(
              char_start, true, alphabet_size, l, alphabet_num_bits, code.len_);
        } else {  // Is rightmost child if bit-prefix of the code is all 1s
          bool is_rightmost_child =
              __popc(code.code_ >> (alphabet_num_bits - l)) == l;

          char_start = getPrevCharStart<T, false>(
              char_start, is_rightmost_child, alphabet_size, l,
              alphabet_num_bits, code.len_);
        }
        if (char_start == 0) {
          start = 0;
        } else {
          T const node_pos = is_pow_two
                                 ? tree.getNodePosAtLevel<true>(char_start, l)
                                 : tree.getNodePosAtLevel<false>(char_start, l);
          if constexpr (ShmemRanks) {
            start = tree.getCounts(char_start) - shmem[ranks_start + node_pos];
          } else {
            start = tree.getCounts(char_start) -
                    tree.getPrecomputedRank(ranks_start + node_pos);
          }
        }
        query.index_ = tree.rank_select_.select<1, ThreadsPerQuery>(
                           l, start + query.index_, offset) +
                       1;
      } else {
        if (char_start == 0) {
          start = 0;
        } else {
          T const node_pos = is_pow_two
                                 ? tree.getNodePosAtLevel<true>(char_start, l)
                                 : tree.getNodePosAtLevel<false>(char_start, l);
          if constexpr (ShmemRanks) {
            start = shmem[ranks_start + node_pos];
          } else {
            start = tree.getPrecomputedRank(ranks_start + node_pos);
          }
        }
        query.index_ = tree.rank_select_.select<0, ThreadsPerQuery>(
                           l, start + query.index_, offset) +
                       1;
      }
      if (l == (code.len_ - 1) and
          query.index_ > tree.rank_select_.bit_array_.size(l)) {
        query.index_ = tree.rank_select_.bit_array_.size(0) + 1;
        break;
      }
      query.index_ -= tree.getCounts(char_start);
      if (l > 1) {
        ranks_start -=
            is_pow_two ? powTwo<T>(l - 1) - 1 : tree.getNumNodesAtLevel(l - 1);
      }
    }
    if (local_t_id == 0) {
      results[i] = query.index_ - 1;  // 0-indexed
    }
  }
}
}  // namespace ecl
