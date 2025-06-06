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

#include <omp.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>

#include <algorithm>
#include <cmath>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <execution>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <vector>

#include "ecl_wavelet/bitarray/rank_select.cuh"
#include "ecl_wavelet/utils/utils.cuh"

// TODO: check if shmem helpful for all kernels
namespace ecl {
typedef unsigned long long int cu_size_t;

/* Thrust host vector that allocates pinned memory */
template <typename T>
using PinnedVector = thrust::host_vector<
    T, thrust::mr::stateless_resource_allocator<
           T, thrust::system::cuda::universal_host_pinned_memory_resource>>;

template <typename T>
class WaveletTree;

namespace detail {
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
 * \brief Empty kernel for creating a graph node.
 */
__global__ LB(MAX_TPB, MIN_BPM) void emptyKernel() {}

/*!
 * \brief Creates a graph for the processing of queries from the host.
 * \param num_chunks Number of chunks the queries will be divided into.
 * \param num_buffers Number of buffers that will be used to store the chunks on
 * the device.
 * \return GraphContents object.
 */
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

/*!
 * \brief Kernel to compute the global histogram of the input data.
 * \details Also replaces the data with the codes or minimal alphabet instead of
 * original symbols.
 * \tparam T Type of the input data.
 * \tparam IsMinAlphabet Whether the alphabet of the input data is minimal.
 * \tparam IsPowTwo Whether the alphabet size is a power of two.
 * \tparam UseShmem Whether to use shared memory for histogram computation.
 * \param tree Wavelet tree.
 * \param data Pointer to input data.
 * \param data_size Number of elements in the input data.
 * \param counts Array to store the counts of the histogram.
 * \param alphabet Alphabet of the input data.
 * \param alphabet_size Size of the alphabet.
 * \param hists_per_block Number of histograms that fit in the shared memory of
 * each block.
 */
template <typename T, bool IsMinAlphabet, bool IsPowTwo, bool UseShmem>
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
    if constexpr (not IsMinAlphabet) {
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
    if constexpr (not IsPowTwo) {
      if (char_data >= tree.getCodesStart()) {
        char_data = tree.encode(char_data).code_;
      }
    }
    if constexpr (not IsMinAlphabet or not IsPowTwo) {
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

/*!
 * \brief Kernel to fill a level of the wavelet tree.
 * \tparam T Type of the input data.
 * \tparam UseShmemPerThread Whether to use shared memory per thread.
 * \param bit_array Bit array the level will be stored in.
 * \param data Data to be filled in the level.
 * \param data_size Size of the data.
 * \param alphabet_start_bit Bit where the alphabet starts. 0 is the LSB.
 * \param level Level to be filled.
 * \param num_threads Total number of threads the kernel is launched with.
 * \param warps_per_block Number of warps per block.
 */
template <typename T, bool UseShmemPerThread>
__global__ LB(MAX_TPB, MIN_BPM) void fillLevelKernel(
    BitArray bit_array, T* const data, size_t const data_size,
    uint8_t const alphabet_start_bit, uint32_t const level,
    size_t const num_threads, uint32_t const warps_per_block) {
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
        if constexpr (utils::kBankSizeBytes < sizeof(T)) {
          elems_to_skip = j / utils::kBanksPerLine;
        } else {
          elems_to_skip =
              j / ((utils::kBankSizeBytes / sizeof(T)) * utils::kBanksPerLine);
        }
        data_slice[j + elems_to_skip] = data[j + i];
      }
      __syncthreads();
      uint32_t word = 0;
      uint32_t const start = threadIdx.x * sizeof(uint32_t) * 8;
      uint32_t const end = min(start + sizeof(uint32_t) * 8, data_size - i);
      elems_to_skip = 0;
      if constexpr (utils::kBankSizeBytes < sizeof(T)) {
        elems_to_skip = start / utils::kBanksPerLine;
      } else {
        elems_to_skip = start / ((utils::kBankSizeBytes / sizeof(T)) *
                                 utils::kBanksPerLine);
      }
      // Start from the end, since LSB is the first bit
      if (end > start) {
        for (uint32_t j = end - 1; j > start; --j) {
          word |= static_cast<uint8_t>(utils::getBit(
              alphabet_start_bit - level, data_slice[j + elems_to_skip]));
          word <<= 1;
        }
        word |= static_cast<uint8_t>(utils::getBit(
            alphabet_start_bit - level, data_slice[start + elems_to_skip]));
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
    uint8_t const local_t_id = threadIdx.x % WS;
    // Each warp processes a block of data
    for (size_t i = global_t_id; i < data_size_rounded; i += num_threads) {
      T code = 0;
      if (i < data_size) {
        code = data[i];
      }

      // Warp vote to all the bits that need to get written to a word
      uint32_t word =
          __ballot_sync(~0, utils::getBit(alphabet_start_bit - level, code));

      if (local_t_id == 0) {
        shared_words[threadIdx.x / WS] = word;
      }
      __syncthreads();
      if (threadIdx.x < warps_per_block) {
        size_t const index = (i - threadIdx.x) + WS * threadIdx.x;
        if (index < data_size) {
          bit_array.writeWordAtBit(level, index, shared_words[threadIdx.x],
                                   offset);
        }
      }
      __syncthreads();
    }
  }
}

/*!
 * \brief Kernel to precompute the ranks of the wavelet tree.
 * \tparam T Type of the input data.
 * \param tree Wavelet tree.
 * \param node_starts Array of node information.
 * \param total_num_nodes Total number of nodes in the tree.
 */
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

    tree.setPrecomputedRank(i, tree.rank_select_.template rank<1, false, 0>(
                                   level, char_counts, offset));
  }
}

}  // namespace detail

/*!
 * \brief Kernel for computing access queries on the wavelet tree.
 * \tparam T Type of the wavelet tree.
 * \tparam ShmemCounts Whether to use shared memory for counts.
 * \tparam ThreadsPerQuery Number of threads per query.
 * \tparam ShmemOffsets Whether to use shared memory for offsets.
 * \tparam ShmemRanks Whether to use shared memory for precomputed ranks.
 * \param tree Wavelet tree.
 * \param indices Indices of the symbols to be accessed.
 * \param num_indices Number of indices.
 * \param results Array to store the accessed symbols.
 * \param alphabet_size Size of the alphabet.
 * \param num_groups Total number of groups of threads. Equates to total number
 * of threads divided by ThreadsPerQuery.
 * \param num_levels Number of levels in the wavelet tree.
 * \param num_ranks Number of precomputed ranks in the wavelet tree.
 */
template <typename T, bool ShmemCounts, int ThreadsPerQuery, bool ShmemOffsets,
          bool ShmemRanks>
__global__ LB(MAX_TPB, MIN_BPM) void accessKernel(
    WaveletTree<T> tree, size_t* const indices, size_t const num_indices,
    T* results, size_t const alphabet_size, uint32_t const num_groups,
    uint8_t const num_levels, T const num_ranks) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type");
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t shmem[];

  if constexpr (ShmemOffsets) {
    for (uint32_t i = threadIdx.x; i < num_levels; i += blockDim.x) {
      shmem[i] = tree.rank_select_.bit_array_.getOffset(i);
    }
    if constexpr (ShmemRanks) {
      for (uint32_t i = threadIdx.x; i < num_ranks; i += blockDim.x) {
        shmem[num_levels + i] = tree.getPrecomputedRank(i);
      }
      if constexpr (ShmemCounts) {
        for (size_t i = threadIdx.x; i < alphabet_size; i += blockDim.x) {
          shmem[num_levels + num_ranks + i] = tree.getCounts(i);
        }
      }
    }
    __syncthreads();
  }

  size_t* counts = ShmemCounts ? shmem + num_levels + num_ranks : nullptr;
  size_t* ranks = ShmemRanks ? shmem + num_levels : nullptr;
  size_t* offsets = ShmemOffsets ? shmem : nullptr;

  size_t const data_size = tree.rank_select_.bit_array_.size(0);

  uint8_t const local_t_id = threadIdx.x % ThreadsPerQuery;
  size_t const global_group_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerQuery;
  bool const is_pow_two = utils::isPowTwo<size_t>(alphabet_size);
  for (size_t i = global_group_id; i < num_indices; i += num_groups) {
    size_t index = indices[i];
    if (index >= data_size) {
      continue;
    }
    T const result = tree.template deviceAccess<ThreadsPerQuery>(
        index, is_pow_two, counts, offsets, ranks);
    if (local_t_id == 0) {
      results[i] = result;
    }
  }
}

/*!
 * \brief Kernel for computing rank queries on the wavelet tree.
 * \tparam T Type of the wavelet tree.
 * \tparam ShmemCounts Whether to use shared memory for counts.
 * \tparam ThreadsPerQuery Number of threads per query.
 * \tparam ShmemOffsets Whether to use shared memory for offsets.
 * \tparam ShmemRanks Whether to use shared memory for precomputed ranks.
 * \param tree Wavelet tree.
 * \param queries Array of rank queries.
 * \param num_queries Number of queries.
 * \param results Array to store the ranks.
 * \param num_groups Total number of groups of threads. Equates to total number
 * of threads divided by ThreadsPerQuery.
 * \param alphabet_size Size of the alphabet.
 * \param num_levels Number of levels in the wavelet tree.
 * \param num_ranks Number of precomputed ranks in the wavelet tree.
 * \param is_pow_two Whether the alphabet size is a power of two.
 *
 */
template <typename T, bool ShmemCounts, int ThreadsPerQuery, bool ShmemOffsets,
          bool ShmemRanks>
__global__ LB(MAX_TPB, MIN_BPM) void rankKernel(
    WaveletTree<T> tree, RankSelectQuery<T>* const queries,
    size_t const num_queries, size_t* const results, uint32_t const num_groups,
    size_t const alphabet_size, uint8_t const num_levels, T const num_ranks,
    bool const is_pow_two) {
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t shmem[];

  if constexpr (ShmemOffsets) {
    for (uint32_t i = threadIdx.x; i < num_levels; i += blockDim.x) {
      shmem[i] = tree.rank_select_.bit_array_.getOffset(i);
    }
    if constexpr (ShmemRanks) {
      for (uint32_t i = threadIdx.x; i < num_ranks; i += blockDim.x) {
        shmem[num_levels + i] = tree.getPrecomputedRank(i);
      }
    }

    if constexpr (ShmemCounts) {
      for (size_t i = threadIdx.x; i < alphabet_size; i += blockDim.x) {
        shmem[num_levels + num_ranks + i] = tree.getCounts(i);
      }
    }
    __syncthreads();
  }

  size_t* counts = ShmemCounts ? shmem + num_levels + num_ranks : nullptr;
  size_t* ranks = ShmemRanks ? shmem + num_levels : nullptr;
  size_t* offsets = ShmemOffsets ? shmem : nullptr;

  uint8_t const local_t_id = threadIdx.x % ThreadsPerQuery;

  size_t const global_group_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerQuery;
  for (size_t i = global_group_id; i < num_queries; i += num_groups) {
    RankSelectQuery<T> query = queries[i];
    if (tree.getTotalAppearances(query.symbol_) == 0) {
      if (local_t_id == 0) {
        results[i] = 0;
      }
      continue;
    }
    size_t const result = tree.template deviceRank<ThreadsPerQuery>(
        query, is_pow_two, counts, offsets, ranks);

    if (local_t_id == 0) {
      results[i] = result;
    }
  }
}

/*!
 * \brief Kernel for computing select queries on the wavelet tree.
 * \tparam T Type of the wavelet tree.
 * \tparam ThreadsPerQuery Number of threads per query.
 * \tparam ShmemRanks Whether to use shared memory for precomputed ranks.
 * \param tree Wavelet tree.
 * \param queries Array of select queries.
 * \param num_queries Number of queries.
 * \param results Array to store the select results.
 * \param num_groups Total number of groups of threads. Equates to total number
 * of threads divided by ThreadsPerQuery.
 * \param is_pow_two Whether the alphabet size is a power of two.
 * \param num_ranks Number of precomputed ranks in the wavelet tree.
 */
template <typename T, int ThreadsPerQuery, bool ShmemRanks>
__global__ LB(MAX_TPB, MIN_BPM) void selectKernel(
    WaveletTree<T> tree, RankSelectQuery<T>* const queries,
    size_t const num_queries, size_t* const results, uint32_t const num_groups,
    bool const is_pow_two, T const num_ranks) {
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t shmem[];
  if constexpr (ShmemRanks) {
    for (uint32_t i = threadIdx.x; i < num_ranks; i += blockDim.x) {
      shmem[i] = tree.getPrecomputedRank(i);
    }
    __syncthreads();
  }

  size_t* ranks = ShmemRanks ? shmem : nullptr;

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
    size_t const result =
        tree.template deviceSelect<ThreadsPerQuery>(query, is_pow_two, ranks);

    if (local_t_id == 0) {
      results[i] = result;
    }
  }
}

/*!
 * \brief Wavelet tree class.
 * \tparam T Type of the text data the tree will be built upon. Must be an
 * unsigned integral type.
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

  /*! Default constructor*/
  WaveletTree() = default;

  /*!
   * \brief Constructor. Builds the wavelet tree from the input data.
   * \throws std::runtime_error if not enough memory is available.
   * \param data Input data to build the wavelet tree.
   * \param data_size Number of elements in the input data.
   * \param alphabet Alphabet of the input data. If not sure of it, pass an
   * empty vector. Must be sorted.
   * \param GPU_index Index of the GPU to use.
   */
  __host__ WaveletTree(T const* data, size_t data_size,
                       std::vector<T>&& alphabet, uint32_t const GPU_index)
      : alphabet_(alphabet), is_copy_(false), GPU_index_(GPU_index) {
    static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                  "T must be an unsigned integral type");
    assert(data_size > 0);
    assert(alphabet_.size() >= utils::kMinAlphabetSize or
           alphabet_.size() == 0);
    assert(alphabet_.size() == 0 or
           std::is_sorted(alphabet_.begin(), alphabet_.end()));

    utils::checkWarpSize(GPU_index_);
    alphabet_size_ = alphabet_.size();

    bool const create_alphabet = alphabet_size_ == 0;
    if (create_alphabet) {
      std::unordered_set<T> alphabet_set;
#pragma omp parallel
      {
        auto const t_id = omp_get_thread_num();
        auto const num_threads = omp_get_num_threads();
        size_t const start = t_id * data_size / num_threads;
        size_t const end = t_id == num_threads - 1
                               ? data_size
                               : (t_id + 1) * data_size / num_threads;
        auto local_set = std::unordered_set<T>();
        for (size_t i = start; i < end; ++i) {
          local_set.insert(data[i]);
        }
#pragma omp critical
        {
          alphabet_set.insert(local_set.begin(), local_set.end());
        }
      }
      alphabet_.assign(alphabet_set.begin(), alphabet_set.end());
      std::sort(std::execution::par, alphabet_.begin(), alphabet_.end());
      alphabet_size_ = alphabet_.size();
      assert(alphabet_size_ >= utils::kMinAlphabetSize);
    }

    bool const is_pow_two = utils::isPowTwo(alphabet_size_);

    // Check if alphabet is already minimal
    is_min_alphabet_ =
        std::all_of(alphabet_.begin(), alphabet_.end(),
                    [i = 0u](unsigned value) mutable { return value == i++; });

    std::vector<Code> codes;
    std::vector<detail::NodeInfo<T>> node_starts;
    // make minimal alphabet
    if (not is_min_alphabet_) {
      auto min_alphabet = std::vector<T>(alphabet_size_);
      std::iota(min_alphabet.begin(), min_alphabet.end(), 0ULL);
      if (not is_pow_two) {
        codes = createMinimalCodes(min_alphabet);
      }
      node_starts = getNodeInfos(min_alphabet, codes);
    } else {
      if (not is_pow_two) {
        codes = createMinimalCodes(alphabet_);
      }
      node_starts = getNodeInfos(alphabet_, codes);
    }

    num_levels_ = utils::ceilLog2(alphabet_size_);
    alphabet_start_bit_ = num_levels_ - 1;
    codes_start_ = alphabet_size_ - codes.size();

    auto const needed_memory =
        WaveletTree::getNeededGPUMemory(data_size, alphabet_size_, num_levels_,
                                        codes.size()) +
        10'000'000 * sizeof(RankSelectQuery<T>);  // Extra memory for queries

    size_t free_memory, total_memory;
    gpuErrchk(cudaMemGetInfo(&free_memory, &total_memory));
    if (needed_memory > free_memory) {
      throw std::runtime_error(
          "Not enough memory available for the wavelet tree.");
    }
    free_memory -= needed_memory;
    bool const use_data_unified_memory = data_size * sizeof(T) > free_memory;
    free_memory -= use_data_unified_memory ? 0 : data_size * sizeof(T);
    bool const use_sorted_data_unified_memory =
        data_size * sizeof(T) > free_memory;
    free_memory -= use_sorted_data_unified_memory ? 0 : data_size * sizeof(T);

    size_t radix_sort_bytes = 0;
    cub::DoubleBuffer<T> d_data_buffer;

    gpuErrchk(cub::DeviceRadixSort::SortKeys(nullptr, radix_sort_bytes,
                                             d_data_buffer, data_size));

    size_t exclusive_sum_bytes = 0;
    gpuErrchk(cub::DeviceScan::ExclusiveSum(nullptr, exclusive_sum_bytes,
                                            (size_t*)data, alphabet_size_));

    auto max_needed_storage = std::max(radix_sort_bytes, exclusive_sum_bytes);
    bool const use_cub_unified_memory = max_needed_storage > free_memory;

    if (use_data_unified_memory or use_sorted_data_unified_memory or
        use_cub_unified_memory) {
      int64_t const free_ram = utils::getAvailableMemoryLinux();
      int64_t needed_ram = 0;
      needed_ram +=
          use_data_unified_memory ? data_size * sizeof(T) : 0;  // d_data
      needed_ram += use_sorted_data_unified_memory ? data_size * sizeof(T)
                                                   : 0;  // d_sorted_data
      needed_ram +=
          use_cub_unified_memory ? max_needed_storage : 0;  // d_temp_storage
      if (needed_ram > free_ram) {
        throw std::runtime_error(
            "Not enough memory available for the wavelet tree.");
      }
    }

    //  Placeholder allocation for bit array to avoid memory fragmentation
    //  problems
    void* d_placeholder;
    gpuErrchk(cudaMalloc(&d_placeholder, num_levels_ * data_size / 8));

    std::vector<T> num_nodes_at_level(num_levels_ - 1);
    //  create codes and copy to device
    if (not is_pow_two) {
      gpuErrchk(cudaMalloc(&d_codes_, codes.size() * sizeof(Code)));
      gpuErrchk(cudaMemcpy(d_codes_, codes.data(), codes.size() * sizeof(Code),
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
      for (uint32_t i = 0; i < num_nodes_at_level.size(); ++i) {
        num_nodes_until_last_level_ += num_nodes_at_level[i];
      }
      gpuErrchk(
          cudaMalloc(&d_num_nodes_at_level_, (num_levels_ - 1) * sizeof(T)));
      gpuErrchk(cudaMemcpy(d_num_nodes_at_level_, num_nodes_at_level.data(),
                           (num_levels_ - 1) * sizeof(T),
                           cudaMemcpyHostToDevice));
    } else {
      num_nodes_until_last_level_ =
          node_starts.size() - (utils::powTwo<T>(num_levels_ - 1) - 1);
    }
    num_ranks_ = node_starts.size();

    gpuErrchk(cudaMalloc(&d_ranks_, num_ranks_ * sizeof(size_t)));

    // Allocate space for counts array
    gpuErrchk(cudaMalloc(&d_counts_, alphabet_size_ * sizeof(size_t)));
    gpuErrchk(cudaMemset(d_counts_, 0, alphabet_size_ * sizeof(size_t)));

    void* d_temp_storage = nullptr;

    if (use_cub_unified_memory) {
      gpuErrchk(cudaMallocManaged(&d_temp_storage, max_needed_storage));
    } else {
      gpuErrchk(cudaMalloc(&d_temp_storage, max_needed_storage));
    }

    T* d_alphabet;
    if (alphabet_size_ * sizeof(T) <= max_needed_storage) {
      // Copy alphabet to device using temp_storage
      d_alphabet = reinterpret_cast<T*>(d_temp_storage);
      gpuErrchk(cudaMemcpy(d_temp_storage, alphabet_.data(),
                           alphabet_size_ * sizeof(T), cudaMemcpyHostToDevice));
    } else {
      gpuErrchk(cudaMalloc(&d_alphabet, alphabet_size_ * sizeof(T)));
      gpuErrchk(cudaMemcpy(d_alphabet, alphabet_.data(),
                           alphabet_size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Copy data to device
    T* d_data;
    if (use_data_unified_memory) {
      gpuErrchk(cudaMallocManaged(&d_data, data_size * sizeof(T)));
    } else {
      gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
    }
    gpuErrchk(cudaMemcpy(d_data, data, data_size * sizeof(T),
                         cudaMemcpyHostToDevice));

    computeGlobalHistogram(is_pow_two, data_size, d_data, d_alphabet,
                           d_counts_);

    // Copy counts to host
    counts_ = std::vector<size_t>(alphabet_size_);
    gpuErrchk(cudaMemcpy(counts_.data(), d_counts_,
                         alphabet_size_ * sizeof(size_t),
                         cudaMemcpyDeviceToHost));

    // Perform exclusive sum of counts
    gpuErrchk(cub::DeviceScan::ExclusiveSum(d_temp_storage, max_needed_storage,
                                            d_counts_, alphabet_size_));

    // Calculate size of bit array at each level
    std::vector<size_t> bit_array_sizes(num_levels_, data_size);
    if (codes.size() > 0) {  // Get min code length
      uint8_t const min_code_len = codes.back().len_;
#pragma omp parallel for
      for (size_t i = num_levels_ - 1; i >= min_code_len; --i) {
        for (int64_t j = alphabet_size_ - codes_start_ - 1; j >= 0; --j) {
          if (i >= codes[j].len_) {
            bit_array_sizes[i] -= counts_[codes_start_ + j];
          } else {
            break;
          }
        }
      }
    }
    std::exclusive_scan(counts_.begin(), counts_.end(), counts_.begin(), 0ULL);

    gpuErrchk(cudaFree(d_placeholder));
    BitArray bit_array(bit_array_sizes, false);

    // Allocate space for sorted data
    T* d_sorted_data;
    if (use_sorted_data_unified_memory) {
      gpuErrchk(cudaMallocManaged(&d_sorted_data, data_size * sizeof(T)));
    } else {
      gpuErrchk(cudaMalloc(&d_sorted_data, data_size * sizeof(T)));
    }

    d_data_buffer = cub::DoubleBuffer<T>(d_data, d_sorted_data);

    for (uint32_t l = 0; l < num_levels_; l++) {
      if (l > 0) {
        // Always sort whole data, but only fill bit array with necessary
        // characters
        gpuErrchk(cub::DeviceRadixSort::SortKeys(
            d_temp_storage, max_needed_storage, d_data_buffer, data_size,
            alphabet_start_bit_ + 1 - l, alphabet_start_bit_ + 1,
            cudaStreamDefault));
      }
      //  Fill l-th bit array
      fillLevel(bit_array, d_data_buffer.Current(), bit_array_sizes[l],
                l);  // synchronous
    }

    if (alphabet_size_ * sizeof(T) > max_needed_storage) {
      gpuErrchk(cudaFreeAsync(d_alphabet, cudaStreamDefault));
    }
    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_sorted_data));
    gpuErrchk(cudaFree(d_temp_storage));

    // build rank and select structures from bit-vectors
    rank_select_ = RankSelect(std::move(bit_array), GPU_index_);

    if (num_ranks_ > 0) {
      detail::NodeInfo<T>* d_node_starts = nullptr;
      gpuErrchk(cudaMallocAsync(&d_node_starts,
                                num_ranks_ * sizeof(detail::NodeInfo<T>),
                                cudaStreamDefault));
      gpuErrchk(cudaMemcpyAsync(d_node_starts, node_starts.data(),
                                num_ranks_ * sizeof(detail::NodeInfo<T>),
                                cudaMemcpyHostToDevice, cudaStreamDefault));
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, detail::precomputeRanksKernel<T>);

      auto const& prop = utils::getDeviceProperties();
      auto const num_warps =
          std::min(static_cast<size_t>((num_ranks_ + WS - 1) / WS),
                   static_cast<size_t>(prop.multiProcessorCount *
                                       prop.maxThreadsPerMultiProcessor / WS));
      auto const [blocks, threads] = utils::getLaunchConfig(
          num_warps, utils::kMinTPB,
          std::min(utils::kMaxTPB,
                   static_cast<uint32_t>(attr.maxThreadsPerBlock)));
      detail::precomputeRanksKernel<T>
          <<<blocks, threads, 0, cudaStreamDefault>>>(*this, d_node_starts,
                                                      num_ranks_);
      gpuErrchk(cudaFreeAsync(d_node_starts, cudaStreamDefault));
    }

    gpuErrchk(cudaHostAlloc(&access_pinned_mem_pool_,
                            access_mem_pool_size_ * sizeof(T),
                            cudaHostAllocPortable));

    gpuErrchk(cudaHostAlloc(&rank_pinned_mem_pool_,
                            rank_mem_pool_size_ * sizeof(size_t),
                            cudaHostAllocPortable));

    gpuErrchk(cudaHostAlloc(&select_pinned_mem_pool_,
                            select_mem_pool_size_ * sizeof(size_t),
                            cudaHostAllocPortable));
  }

  /*! Copy constructor*/
  __host__ WaveletTree(WaveletTree const& other)
      : alphabet_(other.alphabet_),
        rank_select_(other.rank_select_),
        alphabet_size_(other.alphabet_size_),
        alphabet_start_bit_(other.alphabet_start_bit_),
        d_codes_(other.d_codes_),
        d_counts_(other.d_counts_),
        num_levels_(other.num_levels_),
        is_min_alphabet_(other.is_min_alphabet_),
        codes_start_(other.codes_start_),
        access_pinned_mem_pool_(other.access_pinned_mem_pool_),
        rank_pinned_mem_pool_(other.rank_pinned_mem_pool_),
        d_num_nodes_at_level_(other.d_num_nodes_at_level_),
        select_pinned_mem_pool_(other.select_pinned_mem_pool_),
        num_nodes_until_last_level_(other.num_nodes_until_last_level_),
        d_ranks_(other.d_ranks_),
        num_ranks_(other.num_ranks_),
        is_copy_(true) {}

  /*! Deleted copy assignment operator*/
  WaveletTree& operator=(const WaveletTree&) = delete;

  /*! Deleted move constructor*/
  WaveletTree(WaveletTree&&) = delete;

  /*! Deleted move assignment operator*/
  WaveletTree& operator=(WaveletTree&& other) {
    rank_select_ = std::move(other.rank_select_);
    alphabet_ = std::move(other.alphabet_);
    alphabet_size_ = other.alphabet_size_;
    alphabet_start_bit_ = other.alphabet_start_bit_;
    d_codes_ = other.d_codes_;
    other.d_codes_ = nullptr;
    d_counts_ = other.d_counts_;
    other.d_counts_ = nullptr;
    counts_ = std::move(other.counts_);
    num_levels_ = other.num_levels_;
    is_min_alphabet_ = other.is_min_alphabet_;
    codes_start_ = other.codes_start_;
    is_copy_ = other.is_copy_;
    other.is_copy_ = true;
    access_pinned_mem_pool_ = other.access_pinned_mem_pool_;
    other.access_pinned_mem_pool_ = nullptr;
    access_mem_pool_size_ = other.access_mem_pool_size_;
    rank_pinned_mem_pool_ = other.rank_pinned_mem_pool_;
    other.rank_pinned_mem_pool_ = nullptr;
    rank_mem_pool_size_ = other.rank_mem_pool_size_;
    select_pinned_mem_pool_ = other.select_pinned_mem_pool_;
    other.select_pinned_mem_pool_ = nullptr;
    select_mem_pool_size_ = other.select_mem_pool_size_;
    num_nodes_until_last_level_ = other.num_nodes_until_last_level_;
    d_ranks_ = other.d_ranks_;
    other.d_ranks_ = nullptr;
    d_num_nodes_at_level_ = other.d_num_nodes_at_level_;
    other.d_num_nodes_at_level_ = nullptr;
    num_ranks_ = other.num_ranks_;
    GPU_index_ = other.GPU_index_;
    return *this;
  }

  /*! Destructor*/
  ~WaveletTree() {
    if (not is_copy_) {
      gpuErrchk(cudaFree(d_codes_));
      gpuErrchk(cudaFree(d_counts_));
      gpuErrchk(cudaFreeHost(access_pinned_mem_pool_));
      gpuErrchk(cudaFreeHost(rank_pinned_mem_pool_));
      gpuErrchk(cudaFreeHost(select_pinned_mem_pool_));
      if (d_num_nodes_at_level_ != nullptr) {
        gpuErrchk(cudaFree(d_num_nodes_at_level_));
      }
      gpuErrchk(cudaFree(d_ranks_));
    }
  }

  /*!
   * \brief Access the symbols at the given indices in the wavelet tree.
   * \details For best performance, allocate the indices array in pinned
   * memory. For this, the \c PinnedVector class can be used.
   * \throws std::runtime_error if not enough GPU memory is available for the
   * queries. If it happens, try to process the queries in smaller batches.
   * \param indices Pointer to the indices of the symbols to be accessed.
   * \param num_indices Number of indices.
   * \return Access to the memory pool containing the accessed symbols.
   * Processing new access queries may overwrite old ones.
   */
  __host__ [[nodiscard]] std::span<T> access(size_t* indices,
                                             size_t const num_indices) {
    // Number of threads per query, set to 1 since it showed the best
    // performance
    uint8_t constexpr NumThreads = 1;

    assert(num_indices > 0);
    size_t free_mem, total_mem;
    gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));

    size_t constexpr kMaxNumChunks = 10;
    size_t const needed_memory =
        num_indices * sizeof(T) +
        3 * (num_indices / kMaxNumChunks + num_indices % kMaxNumChunks) *
            sizeof(size_t);  // 3 instead of two for having a small buffer
    if (needed_memory > free_mem) {
      throw std::runtime_error(
          "Not enough GPU memory available for the access queries.");
    }

    // CHeck if indices ptr points to pinned memory
    cudaPointerAttributes attr;
    gpuErrchk(cudaPointerGetAttributes(&attr, indices));
    bool const is_pinned_mem = attr.type == cudaMemoryTypeHost;

    std::thread cpu_mem_thread;
    if (not is_pinned_mem) {
      cpu_mem_thread = std::thread([&]() {
        gpuErrchk(cudaHostRegister(indices, num_indices * sizeof(size_t),
                                   cudaHostRegisterPortable));
      });
    }

    // should rarely happen
    if (num_indices > access_mem_pool_size_) {
      gpuErrchk(cudaFreeHost(access_pinned_mem_pool_));
      access_mem_pool_size_ = num_indices;
      gpuErrchk(cudaHostAlloc(&access_pinned_mem_pool_,
                              access_mem_pool_size_ * sizeof(T),
                              cudaHostAllocPortable));
    }

    uint32_t const num_chunks = num_indices < kMaxNumChunks ? 1 : kMaxNumChunks;
    uint8_t const num_buffers = std::min(num_chunks, 2U);
    size_t const chunk_size = num_indices / num_chunks;
    size_t const last_chunk_size = chunk_size + num_indices % num_chunks;

    size_t* d_indices;
    T* d_results;

    bool alloc_failed = false;
    std::thread gpu_alloc_thread([&]() {
      gpuErrchk(cudaSetDevice(GPU_index_));
      cudaMalloc(&d_indices, last_chunk_size * num_buffers * sizeof(size_t));
      auto res = cudaGetLastError();
      if (res != cudaSuccess) {
        alloc_failed = true;
      }

      cudaMalloc(&d_results, num_indices * sizeof(T));
      res = cudaGetLastError();
      if (res != cudaSuccess) {
        alloc_failed = true;
      }
    });

    T* results = access_pinned_mem_pool_;
    struct cudaDeviceProp& prop = utils::getDeviceProperties();

    static bool is_first_call = true;
    static uint32_t max_threads_per_block = 0;
    static size_t static_shmem_per_block = 0;
    if (is_first_call) {
      is_first_call = false;

      struct cudaFuncAttributes funcAttrib;
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, accessKernel<T, true, NumThreads, true, true>));

      max_threads_per_block = std::min(
          utils::kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

      max_threads_per_block =
          utils::findLargestDivisor(utils::kMaxTPB, max_threads_per_block);

      static_shmem_per_block = funcAttrib.sharedSizeBytes;

      gpuErrchk(cudaFuncSetAttribute(
          accessKernel<T, true, NumThreads, true, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - static_shmem_per_block));
      gpuErrchk(cudaFuncSetAttribute(
          accessKernel<T, false, NumThreads, true, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - static_shmem_per_block));
      gpuErrchk(cudaFuncSetAttribute(
          accessKernel<T, false, NumThreads, true, false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - static_shmem_per_block));
    }
    auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;

    size_t const counts_size = sizeof(size_t) * alphabet_size_;

    size_t const offsets_size = sizeof(size_t) * num_levels_;

    size_t const ranks_size = sizeof(size_t) * num_ranks_;

    size_t shmem_per_block = static_shmem_per_block;

    bool const offsets_shmem =
        (offsets_size + shmem_per_block) * utils::kMinBPM <= max_shmem_per_SM;

    auto min_block_size = utils::kMinTPB;
    if (offsets_shmem) {
      shmem_per_block += offsets_size;
      for (uint32_t block_size = max_threads_per_block;
           block_size >= utils::kMinTPB; block_size /= 2) {
        auto const blocks_per_sm = utils::kMinBPM * utils::kMaxTPB / block_size;
        if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
          min_block_size = 2 * block_size;
          break;
        }
      }
    }

    bool const ranks_shmem =
        (ranks_size + shmem_per_block) * utils::kMinBPM <= max_shmem_per_SM;

    if (ranks_shmem) {
      shmem_per_block += ranks_size;
      for (uint32_t block_size = max_threads_per_block;
           block_size >= utils::kMinTPB; block_size /= 2) {
        auto const blocks_per_sm = utils::kMinBPM * utils::kMaxTPB / block_size;
        if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
          min_block_size = 2 * block_size;
          break;
        }
      }
    }

    bool const counts_shmem =
        (counts_size + shmem_per_block) * utils::kMinBPM <= max_shmem_per_SM;

    if (counts_shmem) {
      shmem_per_block += counts_size;
      for (uint32_t block_size = max_threads_per_block;
           block_size >= utils::kMinTPB; block_size /= 2) {
        auto const blocks_per_sm = utils::kMinBPM * utils::kMaxTPB / block_size;
        if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
          min_block_size = 2 * block_size;
          break;
        }
      }
    }

    auto [num_blocks, threads_per_block] = utils::getLaunchConfig(
        std::min((chunk_size * NumThreads + WS - 1) / WS,
                 static_cast<size_t>((prop.maxThreadsPerMultiProcessor *
                                      prop.multiProcessorCount) /
                                     WS)),
        min_block_size, max_threads_per_block);

    uint32_t const num_groups = (num_blocks * threads_per_block) / NumThreads;

    detail::GraphContents graph_contents;
    if (detail::queries_graph_cache.find(num_chunks) ==
        detail::queries_graph_cache.end()) {
      graph_contents = detail::createQueriesGraph(num_chunks, num_buffers);
      detail::queries_graph_cache[num_chunks] = graph_contents;
    } else {
      graph_contents = detail::queries_graph_cache[num_chunks];
    }

    // Change parameters of graph
    void* kernel_params_base[8] = {
        this,         nullptr,         0,
        nullptr,      &alphabet_size_, (uint32_t*)&num_groups,
        &num_levels_, &num_ranks_};

    auto kernel_node_params_base = cudaKernelNodeParams{
        .func = nullptr,
        .gridDim = dim3(num_blocks, 1, 1),
        .blockDim = dim3(threads_per_block, 1, 1),
        .sharedMemBytes =
            static_cast<uint32_t>(shmem_per_block - static_shmem_per_block),
        .kernelParams = nullptr,
        .extra = nullptr};

    if (offsets_shmem) {
      if (ranks_shmem) {
        if (counts_shmem) {
          kernel_node_params_base.func =
              (void*)(&accessKernel<T, true, NumThreads, true, true>);
        } else {
          kernel_node_params_base.func =
              (void*)(&accessKernel<T, false, NumThreads, true, true>);
        }
      } else {
        kernel_node_params_base.func =
            (void*)(&accessKernel<T, false, NumThreads, true, false>);
      }
    } else {
      kernel_node_params_base.func =
          (void*)(&accessKernel<T, false, NumThreads, false, false>);
    }

    // Allocations must be done before setting parameters
    gpu_alloc_thread.join();
    if (not is_pinned_mem) {
      cpu_mem_thread.join();
    }
    if (alloc_failed) {
      if (not is_pinned_mem) {
        gpuErrchk(cudaHostUnregister(indices));
      }
      throw std::runtime_error(
          "Not enough GPU memory available for the access queries.");
    }
    for (uint16_t i = 0; i < num_chunks; ++i) {
      auto const current_chunk_size =
          i == num_chunks - 1 ? last_chunk_size : chunk_size;
      size_t* current_d_indices =
          d_indices + last_chunk_size * (i % num_buffers);

      gpuErrchk(cudaGraphExecMemcpyNodeSetParams1D(
          graph_contents.graph_exec, graph_contents.copy_indices_nodes[i],
          current_d_indices, indices + i * chunk_size,
          current_chunk_size * sizeof(size_t), cudaMemcpyHostToDevice));

      auto kernel_node_params = kernel_node_params_base;
      T* current_d_results = d_results + i * chunk_size;

      void* kernel_params[8] = {kernel_params_base[0],
                                &current_d_indices,
                                current_chunk_size == last_chunk_size
                                    ? (size_t*)&last_chunk_size
                                    : (size_t*)&chunk_size,
                                &current_d_results,
                                kernel_params_base[4],
                                kernel_params_base[5],
                                kernel_params_base[6],
                                kernel_params_base[7]};

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
      gpuErrchk(cudaSetDevice(GPU_index_));
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

  /*!
   * \brief Rank queries on the wavelet tree. Number of occurrences of a
   * symbol up to a given index (exclusive).
   * \details For best performance, allocate the queries array in pinned
   * memory. For this, the \c PinnedVector class can be used. Creating the
   * queries sorted by symbol can also speed up the processing.
   * \throws std::runtime_error if not enough GPU memory is available for the
   * queries. If it happens, try to process the queries in smaller batches.
   * \param queries Array of rank queries. For best performance, allocate the
   * queries array in pinned memory.
   * \param num_queries Number of queries.
   * \return Access to the memory pool containing the rank results.
   * Processing new rank queries may overwrite old ones.
   */
  __host__ [[nodiscard]] std::span<size_t> rank(RankSelectQuery<T>* queries,
                                                size_t const num_queries) {
    // Number of threads per query, set to 1 since it showed the best
    // performance
    uint8_t constexpr NumThreads = 1;

    assert(std::all_of(queries, queries + num_queries,
                       [&](const RankSelectQuery<T>& s) {
                         return s.index_ < rank_select_.bit_array_.size(0);
                       }));

    size_t free_mem, total_mem;
    gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));

    size_t constexpr kMaxNumChunks = 10;
    size_t const needed_memory =
        num_queries * sizeof(size_t) +
        3 * (num_queries / kMaxNumChunks + num_queries % kMaxNumChunks) *
            sizeof(RankSelectQuery<T>);  // 3 instead of two for having a
                                         // small buffer
    if (needed_memory > free_mem) {
      throw std::runtime_error(
          "Not enough GPU memory available for the rank queries.");
    }

    cudaPointerAttributes attr;
    gpuErrchk(cudaPointerGetAttributes(&attr, queries));
    bool const is_pinned_mem = attr.type == cudaMemoryTypeHost;

    std::thread cpu_mem_thread;
    if (not is_pinned_mem) {
      cpu_mem_thread = std::thread([&]() {
        gpuErrchk(cudaHostRegister(queries,
                                   num_queries * sizeof(RankSelectQuery<T>),
                                   cudaHostRegisterPortable));
      });
    }

    // should rarely happen
    if (num_queries > rank_mem_pool_size_) {
      gpuErrchk(cudaFreeHost(rank_pinned_mem_pool_));
      rank_mem_pool_size_ = num_queries;
      gpuErrchk(cudaHostAlloc(&rank_pinned_mem_pool_,
                              rank_mem_pool_size_ * sizeof(size_t),
                              cudaHostAllocPortable));
    }

    //  Divide indices into chunks
    uint32_t const num_chunks = num_queries < kMaxNumChunks ? 1 : kMaxNumChunks;
    uint8_t const num_buffers = std::min(num_chunks, 2U);
    size_t const chunk_size = num_queries / num_chunks;
    size_t const last_chunk_size = chunk_size + num_queries % num_chunks;

    RankSelectQuery<T>* d_queries;
    size_t* d_results;

    bool alloc_failed = false;
    std::thread gpu_alloc_thread([&]() {
      gpuErrchk(cudaSetDevice(GPU_index_));
      cudaMalloc(&d_queries,
                 last_chunk_size * num_buffers * sizeof(RankSelectQuery<T>));
      auto res = cudaGetLastError();

      if (res != cudaSuccess) {
        alloc_failed = true;
      }

      cudaMalloc(&d_results, num_queries * sizeof(size_t));
      res = cudaGetLastError();
      if (res != cudaSuccess) {
        alloc_failed = true;
      }
    });

    size_t* results = rank_pinned_mem_pool_;
    struct cudaDeviceProp& prop = utils::getDeviceProperties();

    static bool is_first_call = true;
    static uint32_t max_threads_per_block = 0;
    static size_t static_shmem_per_block = 0;

    if (is_first_call) {
      is_first_call = false;

      struct cudaFuncAttributes funcAttrib;
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, rankKernel<T, true, NumThreads, true, true>));
      max_threads_per_block = std::min(
          utils::kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

      max_threads_per_block =
          utils::findLargestDivisor(utils::kMaxTPB, max_threads_per_block);

      static_shmem_per_block = funcAttrib.sharedSizeBytes;

      gpuErrchk(cudaFuncSetAttribute(
          rankKernel<T, true, NumThreads, true, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - static_shmem_per_block));
      gpuErrchk(cudaFuncSetAttribute(
          rankKernel<T, false, NumThreads, true, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - static_shmem_per_block));
      gpuErrchk(cudaFuncSetAttribute(
          rankKernel<T, true, NumThreads, true, false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - static_shmem_per_block));
      gpuErrchk(cudaFuncSetAttribute(
          rankKernel<T, false, NumThreads, true, false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - static_shmem_per_block));
    }

    size_t const counts_size = sizeof(size_t) * alphabet_size_;

    size_t const offsets_size = sizeof(size_t) * num_levels_;

    size_t const ranks_size = sizeof(size_t) * num_ranks_;

    auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;

    size_t shmem_per_block = static_shmem_per_block;

    bool const offsets_shmem =
        (offsets_size + shmem_per_block) * utils::kMinBPM <= max_shmem_per_SM;

    auto min_block_size = utils::kMinTPB;
    if (offsets_shmem) {
      shmem_per_block += offsets_size;
      for (uint32_t block_size = max_threads_per_block;
           block_size >= utils::kMinTPB; block_size /= 2) {
        auto const blocks_per_sm = utils::kMinBPM * utils::kMaxTPB / block_size;
        if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
          min_block_size = 2 * block_size;
          break;
        }
      }
    }

    bool const ranks_shmem =
        (ranks_size + shmem_per_block) * utils::kMinBPM <= max_shmem_per_SM;

    if (ranks_shmem) {
      shmem_per_block += ranks_size;
      for (uint32_t block_size = max_threads_per_block;
           block_size >= utils::kMinTPB; block_size /= 2) {
        auto const blocks_per_sm = utils::kMinBPM * utils::kMaxTPB / block_size;
        if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
          min_block_size = 2 * block_size;
          break;
        }
      }
    }

    bool const counts_shmem =
        (counts_size + shmem_per_block) * utils::kMinBPM <= max_shmem_per_SM;

    if (counts_shmem) {
      shmem_per_block += counts_size;
      for (uint32_t block_size = max_threads_per_block;
           block_size >= utils::kMinTPB; block_size /= 2) {
        auto const blocks_per_sm = utils::kMinBPM * utils::kMaxTPB / block_size;
        if (shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
          min_block_size = 2 * block_size;
          break;
        }
      }
    }

    auto [num_blocks, threads_per_block] = utils::getLaunchConfig(
        std::min((chunk_size * NumThreads + WS - 1) / WS,
                 static_cast<size_t>((prop.maxThreadsPerMultiProcessor *
                                      prop.multiProcessorCount) /
                                     WS)),
        min_block_size, max_threads_per_block);

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

    detail::GraphContents graph_contents;
    if (detail::queries_graph_cache.find(num_chunks) ==
        detail::queries_graph_cache.end()) {
      graph_contents = detail::createQueriesGraph(num_chunks, num_buffers);
      detail::queries_graph_cache[num_chunks] = graph_contents;
    } else {
      graph_contents = detail::queries_graph_cache[num_chunks];
    }

    bool is_pow_two = utils::isPowTwo(alphabet_size_);
    // Change parameters of graph
    void* kernel_params_base[9] = {this,
                                   nullptr,
                                   0,
                                   nullptr,
                                   (uint32_t*)&num_groups,
                                   &alphabet_size_,
                                   &num_levels_,
                                   &num_ranks_,
                                   &is_pow_two};

    auto kernel_node_params_base = cudaKernelNodeParams{
        .func = nullptr,
        .gridDim = dim3(num_blocks, 1, 1),
        .blockDim = dim3(threads_per_block, 1, 1),
        .sharedMemBytes =
            static_cast<uint32_t>(shmem_per_block - static_shmem_per_block),
        .kernelParams = nullptr,
        .extra = nullptr};

    if (offsets_shmem) {
      if (ranks_shmem) {
        if (counts_shmem) {
          kernel_node_params_base.func =
              (void*)(&rankKernel<T, true, NumThreads, true, true>);
        } else {
          kernel_node_params_base.func =
              (void*)(&rankKernel<T, false, NumThreads, true, true>);
        }
      } else {
        if (counts_shmem) {
          kernel_node_params_base.func =
              (void*)(&rankKernel<T, true, NumThreads, true, false>);
        } else {
          kernel_node_params_base.func =
              (void*)(&rankKernel<T, false, NumThreads, true, false>);
        }
      }
    } else {
      kernel_node_params_base.func =
          (void*)(&rankKernel<T, false, NumThreads, false, false>);
    }

    // Allocations must be done before setting parameters
    gpu_alloc_thread.join();
    if (not is_pinned_mem) {
      cpu_mem_thread.join();
    }
    if (alloc_failed) {
      if (not is_pinned_mem) {
        gpuErrchk(cudaHostUnregister(queries));
      }
      throw std::runtime_error(
          "Not enough GPU memory available for the rank queries.");
    }
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

      void* kernel_params[9] = {kernel_params_base[0],
                                &current_d_queries,
                                current_chunk_size == last_chunk_size
                                    ? (size_t*)&last_chunk_size
                                    : (size_t*)&chunk_size,
                                &current_d_results,
                                kernel_params_base[4],
                                kernel_params_base[5],
                                kernel_params_base[6],
                                kernel_params_base[7],
                                kernel_params_base[8]};

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

  /*!
   * \brief Select queries on the wavelet tree. Find the index of the k-th
   * occurrence of a symbol. Starts counting from 1.
   * \details For best performance, allocate the queries array in pinned
   * memory. For this, the \c PinnedVector class can be used. Creating the
   * queries sorted by symbol can also speed up the processing.
   * \throws std::runtime_error if not enough GPU memory is available for the
   * queries. If it happens, try to process the queries in smaller batches.
   * \param queries Array of select queries. For best performance, allocate
   * the queries array in pinned memory.
   * \param num_queries Number of queries.
   * \return Access to the memory pool containing the select results.
   * Processing new select queries may overwrite old ones.
   */
  __host__ [[nodiscard]] std::span<size_t> select(RankSelectQuery<T>* queries,
                                                  size_t const num_queries) {
    // Number of threads per query, set to 1 since it showed the best
    // performance
    uint8_t constexpr ThreadsPerQuery = 1;

    assert(
        std::all_of(queries, queries + num_queries,
                    [](const RankSelectQuery<T>& s) { return s.index_ > 0; }));

    size_t free_mem, total_mem;
    gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));

    size_t constexpr kMaxNumChunks = 40;
    size_t const needed_memory =
        num_queries * sizeof(size_t) +
        3 * (num_queries / kMaxNumChunks + num_queries % kMaxNumChunks) *
            sizeof(RankSelectQuery<T>);  // 3 instead of two for having a
                                         // small buffer
    if (needed_memory > free_mem) {
      throw std::runtime_error(
          "Not enough GPU memory available for the select queries.");
    }

    cudaPointerAttributes attr;
    gpuErrchk(cudaPointerGetAttributes(&attr, queries));
    bool const is_pinned_mem = attr.type == cudaMemoryTypeHost;

    std::thread cpu_mem_thread;
    if (not is_pinned_mem) {
      cpu_mem_thread = std::thread([&]() {
        gpuErrchk(cudaHostRegister(queries,
                                   num_queries * sizeof(RankSelectQuery<T>),
                                   cudaHostRegisterPortable));
      });
    }

    // should rarely happen
    if (num_queries > select_mem_pool_size_) {
      gpuErrchk(cudaFreeHost(select_pinned_mem_pool_));
      select_mem_pool_size_ = num_queries;
      gpuErrchk(cudaHostAlloc(&select_pinned_mem_pool_,
                              select_mem_pool_size_ * sizeof(size_t),
                              cudaHostAllocPortable));
    }

    //  Divide indices into chunks
    uint32_t const num_chunks = num_queries < kMaxNumChunks ? 1 : kMaxNumChunks;
    uint8_t const num_buffers = std::min(num_chunks, 2U);
    size_t const chunk_size = num_queries / num_chunks;
    size_t const last_chunk_size = chunk_size + num_queries % num_chunks;

    RankSelectQuery<T>* d_queries;
    size_t* d_results;

    bool alloc_failed = false;
    std::thread gpu_alloc_thread([&]() {
      gpuErrchk(cudaSetDevice(GPU_index_));
      cudaMalloc(&d_queries,
                 last_chunk_size * num_buffers * sizeof(RankSelectQuery<T>));
      auto res = cudaGetLastError();

      if (res != cudaSuccess) {
        alloc_failed = true;
      }

      cudaMalloc(&d_results, num_queries * sizeof(size_t));
      res = cudaGetLastError();
      if (res != cudaSuccess) {
        alloc_failed = true;
      }
    });

    size_t* results = select_pinned_mem_pool_;
    auto& prop = utils::getDeviceProperties();

    static bool is_first_call = true;
    static uint32_t max_threads_per_block = 0;
    static size_t static_shmem_per_block = 0;

    if (is_first_call) {
      is_first_call = false;

      struct cudaFuncAttributes funcAttrib;
      gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                      selectKernel<T, ThreadsPerQuery, true>));
      max_threads_per_block = std::min(
          utils::kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

      max_threads_per_block =
          utils::findLargestDivisor(utils::kMaxTPB, max_threads_per_block);

      static_shmem_per_block = funcAttrib.sharedSizeBytes;
      gpuErrchk(cudaFuncSetAttribute(
          selectKernel<T, ThreadsPerQuery, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - static_shmem_per_block));
    }

    size_t const ranks_size = sizeof(size_t) * num_ranks_;

    size_t const total_shmem_per_block = ranks_size + static_shmem_per_block;

    auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;

    bool const ranks_shmem =
        total_shmem_per_block * utils::kMinBPM <= max_shmem_per_SM;

    auto min_block_size = utils::kMinTPB;
    if (ranks_shmem) {
      for (uint32_t block_size = max_threads_per_block;
           block_size >= utils::kMinTPB; block_size /= 2) {
        auto const blocks_per_sm = utils::kMinBPM * utils::kMaxTPB / block_size;
        if (total_shmem_per_block * blocks_per_sm > max_shmem_per_SM) {
          min_block_size = 2 * block_size;
          break;
        }
      }
    }

    auto [num_blocks, threads_per_block] = utils::getLaunchConfig(
        std::min((chunk_size * ThreadsPerQuery + WS - 1) / WS,
                 static_cast<size_t>((prop.maxThreadsPerMultiProcessor *
                                      prop.multiProcessorCount) /
                                     WS)),
        min_block_size, max_threads_per_block);

    uint32_t const num_groups =
        (num_blocks * threads_per_block) / ThreadsPerQuery;

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

    detail::GraphContents graph_contents;
    if (detail::queries_graph_cache.find(num_chunks) ==
        detail::queries_graph_cache.end()) {
      graph_contents = detail::createQueriesGraph(num_chunks, num_buffers);
      detail::queries_graph_cache[num_chunks] = graph_contents;
    } else {
      graph_contents = detail::queries_graph_cache[num_chunks];
    }
    bool is_pow_two = utils::isPowTwo(alphabet_size_);
    // Change parameters of graph
    void* kernel_params_base[7] = {
        this,        nullptr,    0, nullptr, (uint32_t*)&num_groups,
        &is_pow_two, &num_ranks_};

    auto kernel_node_params_base =
        cudaKernelNodeParams{.func = nullptr,
                             .gridDim = dim3(num_blocks, 1, 1),
                             .blockDim = dim3(threads_per_block, 1, 1),
                             .sharedMemBytes = 0,
                             .kernelParams = nullptr,
                             .extra = nullptr};
    if (ranks_shmem) {
      kernel_node_params_base.func =
          (void*)(&selectKernel<T, ThreadsPerQuery, true>);
      kernel_node_params_base.sharedMemBytes = ranks_size;
    } else {
      kernel_node_params_base.func =
          (void*)(&selectKernel<T, ThreadsPerQuery, false>);
    }

    gpu_alloc_thread.join();
    if (not is_pinned_mem) {
      cpu_mem_thread.join();
    }
    if (alloc_failed) {
      if (not is_pinned_mem) {
        gpuErrchk(cudaHostUnregister(queries));
      }
      throw std::runtime_error(
          "Not enough GPU memory available for the select queries.");
    }
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

      void* kernel_params[7] = {kernel_params_base[0],
                                &current_d_queries,
                                current_chunk_size == last_chunk_size
                                    ? (size_t*)&last_chunk_size
                                    : (size_t*)&chunk_size,
                                &current_d_results,
                                kernel_params_base[4],
                                kernel_params_base[5],
                                kernel_params_base[6]};

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

  /*!
   * \brief Access the symbol at a given index in the wavelet tree.
   * \tparam ThreadsPerQuery Number of threads per query. Must be a divisor
   * of 32. The best performance is obtained with 1 thread per query, given
   * enough parallelism. More than 8 should be avoided since the main work is
   * a loop that has at most 8 iterations.
   * \param index Index of the symbol to access.
   * \param is_pow_two Whether the alphabet size is a power of two.
   * \param counts Pointer to the shared memory counts array. If nullptr, the
   * default counts array is used.
   * \param offsets Pointer to the shared memory offsets array. If nullptr,
   * the default offsets array is used.
   * \param ranks Pointer to the shared memory precomputed ranks array. If
   * nullptr, the default ranks array is used.
   * \return The symbol at the given index.
   */
  template <int ThreadsPerQuery>
  __device__ [[nodiscard]] T deviceAccess(size_t index, bool const is_pow_two,
                                          size_t* counts = nullptr,
                                          size_t* offsets = nullptr,
                                          size_t* ranks = nullptr) {
    static_assert(
        ThreadsPerQuery > 0 and ThreadsPerQuery <= 32 and
            (32 % ThreadsPerQuery) == 0,
        "ThreadsPerQuery must be a divisor of 32 and greater than 0.");
    if (counts == nullptr) {
      counts = d_counts_;
    }
    if (ranks == nullptr) {
      ranks = d_ranks_;
    }
    T char_start = 0;
    T char_end = alphabet_size_ - 1;
    T ranks_start = 0;
    for (uint8_t l = 0; l < num_levels_; ++l) {
      size_t const char_counts = counts[char_start];
      size_t offset;
      if (offsets == nullptr) {
        offset = rank_select_.bit_array_.getOffset(l);
      } else {
        offset = offsets[l];
      }
      if (char_end - char_start < 2) {
        if (char_end - char_start > 0 and
            rank_select_.bit_array_.access(l, char_counts + index, offset) ==
                true) {
          char_start++;
        }
        break;
      }
      size_t start;
      if (char_start == 0) {
        start = 0;
      } else {
        auto const node_pos = is_pow_two
                                  ? getNodePosAtLevel<true>(char_start, l)
                                  : getNodePosAtLevel<false>(char_start, l);
        start = ranks[ranks_start + node_pos];
      }
      RankResult const result =
          rank_select_.template rank<ThreadsPerQuery, true, 0>(
              l, char_counts + index, offset);
      size_t const pos = result.rank;
      bool const bit_at_index = result.bit;
      // To avoid overflow if alphabet size is the maximum value of T
      T const diff = utils::getPrevPowTwo<size_t>(
          static_cast<size_t>(char_end - char_start) + 1);
      if (bit_at_index == false) {
        index = pos - start;
        char_end = char_start + (diff - 1);
      } else {
        index -= pos - start;
        char_start += diff;
      }
      ranks_start +=
          is_pow_two ? utils::powTwo<T>(l) - 1 : getNumNodesAtLevel(l);
    }
    return char_start;
  }

  /*!
   * \brief Rank query on the wavelet tree. Number of occurrences of a symbol
   * up to a given index (exclusive).
   * \tparam ThreadsPerQuery Number of threads per query. Must be a divisor
   * of 32. The best performance is obtained with 1 thread per query, given
   * enough parallelism. More than 8 should be avoided since the main work is
   * a loop that has at most 8 iterations.
   * \param query Rank query.
   * \param is_pow_two Whether the alphabet size is a power of two.
   * \param counts Pointer to the shared memory counts array. If nullptr, the
   * default counts array is used.
   * \param offsets Pointer to the shared memory offsets array. If nullptr,
   * the default offsets array is used.
   * \param ranks Pointer to the shared memory precomputed ranks array. If
   * nullptr, the default ranks array is used.
   * \return The rank result.
   */
  template <int ThreadsPerQuery>
  __device__ [[nodiscard]] size_t deviceRank(RankSelectQuery<T> const& query,
                                             bool const is_pow_two,
                                             size_t* counts = nullptr,
                                             size_t* offsets = nullptr,
                                             size_t* ranks = nullptr) {
    static_assert(
        ThreadsPerQuery > 0 and ThreadsPerQuery <= 32 and
            (32 % ThreadsPerQuery) == 0,
        "ThreadsPerQuery must be a divisor of 32 and greater than 0.");
    if (counts == nullptr) {
      counts = d_counts_;
    }
    if (ranks == nullptr) {
      ranks = d_ranks_;
    }
    T char_start = 0;
    T char_end = alphabet_size_ - 1;
    T char_split;
    size_t start, pos;
    T ranks_start = 0;

    size_t result = query.index_;
    for (uint8_t l = 0; l < num_levels_; ++l) {
      if (char_end - char_start == 0) {
        break;
      }
      size_t const char_counts = counts[char_start];
      size_t offset;
      if (offsets == nullptr) {
        offset = rank_select_.bit_array_.getOffset(l);
      } else {
        offset = offsets[l];
      }
      if (char_start == 0) {
        start = 0;
      } else {
        auto const node_pos = is_pow_two
                                  ? getNodePosAtLevel<true>(char_start, l)
                                  : getNodePosAtLevel<false>(char_start, l);
        start = ranks[ranks_start + node_pos];
      }
      pos = rank_select_.template rank<ThreadsPerQuery, false, 0>(
          l, char_counts + result, offset);
      char_split =
          char_start + utils::getPrevPowTwo<size_t>(
                           static_cast<size_t>(char_end - char_start) + 1);
      if (query.symbol_ < char_split) {
        result = pos - start;
        char_end = char_split - 1;
      } else {
        result -= pos - start;
        char_start = char_split;
      }
      if (l < num_levels_ - 1) {
        ranks_start +=
            is_pow_two ? utils::powTwo<T>(l) - 1 : getNumNodesAtLevel(l);
      }
    }
    return result;
  }

  /*!
   * \brief Get the start of the node containing a given character at a level.
   * \tparam IsPowTwo Whether the alphabet size is a power of two.
   * \param char_start Character for which to get the start.
   * \param is_rightmost_child Whether the character is in the rightmost node
   * of the level, i.e. all the bits up until that level are 1.
   * \param alphabet_size Size of the alphabet.
   * \param level Level of the node.
   * \param num_levels Number of levels in the wavelet tree.
   * \param code_len Length of the code.
   * \return The start of the node containing the character.
   */
  template <bool IsPowTwo>
  __device__ [[nodiscard]] T getPrevCharStart(T const char_start,
                                              bool const is_rightmost_child,
                                              size_t const alphabet_size,
                                              uint8_t const level,
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
          new_start += utils::getPrevPowTwo(alphabet_size - new_start);
        }
        return new_start;
      } else {
        return char_start - utils::powTwo<uint32_t>(code_len - 1 - level);
      }
    }
  }

  /*!
   * \brief Select query on the wavelet tree. Find the index of the k-th
   * occurrence of a symbol. Starts counting from 1.
   * \tparam ThreadsPerQuery Number of threads per query. Must be a divisor
   * of 32. The best performance is obtained with 1 thread per query, given
   * enough parallelism. More than 8 should be avoided since the main work is
   * a loop that has at most 8 iterations.
   * \param query Select query.
   * \param is_pow_two Whether the alphabet size is a power of two.
   * \param ranks Pointer to the shared memory precomputed ranks array. If
   * nullptr, the default ranks array is used.
   * \return The select result.
   */
  template <int ThreadsPerQuery>
  __device__ [[nodiscard]] size_t deviceSelect(RankSelectQuery<T> const& query,
                                               bool const is_pow_two,
                                               size_t* ranks = nullptr) {
    if (ranks == nullptr) {
      ranks = d_ranks_;
    }

    size_t result = query.index_;
    typename WaveletTree<T>::Code code{num_levels_, query.symbol_};
    if (not is_pow_two and query.symbol_ >= codes_start_) {
      code = encode(query.symbol_);
    }

    T char_start = query.symbol_;
    size_t start;
    T ranks_start = num_nodes_until_last_level_;
    int16_t l = code.len_ - 1;
    for (int16_t level = num_levels_ - 2; level >= l; --level) {
      ranks_start -=
          is_pow_two ? utils::powTwo<T>(level) - 1 : getNumNodesAtLevel(level);
    }
    for (; l >= 0; --l) {
      size_t const offset = rank_select_.bit_array_.getOffset(l);
      // If it's a right child
      if (utils::getBit<T>(num_levels_ - 1 - l, code.code_) == true) {
        if (is_pow_two) {
          char_start = getPrevCharStart<true>(char_start, true, alphabet_size_,
                                              l, num_levels_, code.len_);
        } else {  // Is rightmost child if bit-prefix of the code is all 1s
          bool is_rightmost_child =
              __popc(code.code_ >> (num_levels_ - l)) == l;

          char_start = getPrevCharStart<false>(char_start, is_rightmost_child,
                                               alphabet_size_, l, num_levels_,
                                               code.len_);
        }
        if (char_start == 0) {
          start = 0;
        } else {
          T const node_pos = is_pow_two
                                 ? getNodePosAtLevel<true>(char_start, l)
                                 : getNodePosAtLevel<false>(char_start, l);
          start = getCounts(char_start) - ranks[ranks_start + node_pos];
        }
        result = rank_select_.template select<1, ThreadsPerQuery>(
                     l, start + result, offset) +
                 1;  // 1-indexed
      } else {
        if (char_start == 0) {
          start = 0;
        } else {
          T const node_pos = is_pow_two
                                 ? getNodePosAtLevel<true>(char_start, l)
                                 : getNodePosAtLevel<false>(char_start, l);
          start = ranks[ranks_start + node_pos];
        }
        result = rank_select_.template select<0, ThreadsPerQuery>(
                     l, start + result, offset) +
                 1;  // 1-indexed
      }
      if (l == (code.len_ - 1) and result > rank_select_.bit_array_.size(l)) {
        result = rank_select_.bit_array_.size(0) + 1;
        break;
      }
      result -= getCounts(char_start);
      if (l > 1) {
        ranks_start -= is_pow_two ? utils::powTwo<T>(l - 1) - 1
                                  : getNumNodesAtLevel(l - 1);
      }
    }
    return result - 1;  // 0-indexed
  }

  /*!
   * \brief Encodes a symbol from the alphabet. Only symbols that have a code
   * should be passed as argument.
   * \param c Symbol of the minimal alphabet to be encoded.
   * \return Encoded symbol.
   */
  __device__ [[nodiscard]] Code encode(T const c) const noexcept {
    assert(c < alphabet_size_);
    assert(c >= codes_start_);
    return d_codes_[c - codes_start_];
  }

  /*!
   * \brief Creates minimal codes for the alphabet. Codes are only created for
   * the symbols that are bigger than the previous power of two of the
   * alphabet size.
   * \param alphabet Alphabet to create codes for.
   * \return Vector of codes.
   */
  __host__ [[nodiscard]] static std::vector<Code> createMinimalCodes(
      std::vector<T> const& alphabet) noexcept {
    auto const alphabet_size = alphabet.size();
    if (utils::isPowTwo<size_t>(alphabet_size)) {
      return std::vector<Code>(0);
    }
    size_t total_num_codes = 0;
    std::vector<Code> codes(alphabet_size);
    uint8_t const total_num_bits = utils::ceilLog2<size_t>(alphabet_size);
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
        auto pow_two = utils::powTwo<uint32_t>(i);
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
        codes[alphabet_size - 1].code_ =
            ((1UL << start_bit) - 1) << (alphabet_start_bit + 1 - start_bit);
      } else {
        code_len = utils::ceilLog2<T>(num_codes);
#pragma omp parallel for
        for (size_t i = alphabet_size - num_codes; i < alphabet_size; i++) {
          // Code of local subtree
          T code = i - start_i;
          // Shift code to start at start_bit
          code <<= (total_num_bits - start_bit - code_len);
          // Add to global code already saved
          code +=
              (~((1UL << (total_num_bits - start_bit)) - 1)) & codes[i].code_;

          codes[i].code_ = code;
          codes[i].len_ = start_bit + code_len;
        }
      }
    } while (code_len > 1);
    // Keep only last alphabet_size - codes_start codes
    codes.erase(codes.begin(), codes.begin() + alphabet_size - total_num_codes);
    return codes;
  }

  /*!
   * \brief Get the size of the alphabet.
   * \return Size of the alphabet.
   */
  __host__ __device__ [[nodiscard]] size_t getAlphabetSize() const noexcept {
    return alphabet_size_;
  }

  /*!
   * \brief Get a copy the alphabet.
   * \return Alphabet.
   */
  __host__ std::vector<T> const& getAlphabet() const noexcept {
    return alphabet_;
  }

  /*!
   * \brief Get the number of levels in the wavelet tree.
   * \return Number of levels.
   */
  __device__ [[nodiscard]] uint8_t getNumLevels() const noexcept {
    return num_levels_;
  }

  /*!
   * \brief Get the number of occurrences of all symbols that are
   * lexicographically smaller than the i-th symbol in the alphabet. Starting
   * from 0.
   * \param i Index of the symbol in the alphabet.
   * \return Number of occurrences of all symbols that are lexicographically
   * smaller.
   */
  __host__ __device__ [[nodiscard]] size_t getCounts(size_t i) const noexcept {
    assert(i < alphabet_size_);
#if defined(__CUDA_ARCH__)
    return d_counts_[i];
#else
    return counts_[i];
#endif
  }

  /*!
   * \brief Get the number of occurrences of a symbol.
   * \param i Index of the symbol in the alphabet.
   * \return Number of occurrences of the i-th symbol in the alphabet.
   */
  __host__ __device__ [[nodiscard]] size_t getTotalAppearances(
      size_t i) const noexcept {
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
  __host__ __device__ [[nodiscard]] bool isMinAlphabet() const noexcept {
    return is_min_alphabet_;
  }

  /*!
   * \brief Get the start of the codes in the alphabet.
   * \return Start of the codes.
   */
  __device__ [[nodiscard]] T getCodesStart() const noexcept {
    return codes_start_;
  }

  /*!
   * \brief Get the position of a node at a given level, i.e. whether it is
   * the first, second...
   * \tparam IsPowTwo Whether the alphabet size is a power of two.
   * \param symbol Symbol that marks the start of the node.
   * \param level Level of the node.
   * \return Position of the node at the given level. Zero if it is the second
   * node of the level.
   */
  template <bool IsPowTwo>
  __device__ [[nodiscard]] T getNodePosAtLevel(
      T symbol, uint8_t const level) const noexcept {
    if (level == 1) {
      return 0;
    }
    T num_prev_nodes = 0;
    T node_lens = 1;
    if constexpr (IsPowTwo) {
      // All node lens are equal
      node_lens = 1ULL << (num_levels_ - level);
    } else {
      size_t remaining_symbols = alphabet_size_;
      for (int16_t i = level - 1; i >= 0; --i) {
        auto const subtree_size =
            utils::getPrevPowTwo<size_t>(remaining_symbols);
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

  __device__ [[nodiscard]] size_t getPrecomputedRank(T const i) const noexcept {
    return d_ranks_[i];
  }

  __device__ void setPrecomputedRank(T const i, size_t const rank) noexcept {
    d_ranks_[i] = rank;
  }

  __device__ [[nodiscard]] T getNumNodesAtLevel(
      uint8_t const level) const noexcept {
    assert(level < num_levels_ - 1);
    return d_num_nodes_at_level_[level];
  }

 protected:
  std::vector<T> alphabet_; /*!< Alphabet of the wavelet tree*/

  /*!
   * \brief Compute the global histogram of the data.
   * \param is_pow_two Whether the alphabet size is a power of two.
   * \param data_size Size of the data.
   * \param d_data Pointer to the data on the device.
   * \param d_alphabet Pointer to the alphabet on the device.
   * \param d_histogram Pointer to the histogram on the device.
   */
  __host__ void computeGlobalHistogram(bool const is_pow_two,
                                       size_t const data_size, T* d_data,
                                       T* d_alphabet,
                                       size_t* d_histogram) const noexcept {
    struct cudaFuncAttributes funcAttrib;
    if (is_pow_two) {
      if (is_min_alphabet_) {
        gpuErrchk(cudaFuncGetAttributes(
            &funcAttrib,
            detail::computeGlobalHistogramKernel<T, true, true, true>));
      } else {
        gpuErrchk(cudaFuncGetAttributes(
            &funcAttrib,
            detail::computeGlobalHistogramKernel<T, false, true, true>));
      }
    } else {
      if (is_min_alphabet_) {
        gpuErrchk(cudaFuncGetAttributes(
            &funcAttrib,
            detail::computeGlobalHistogramKernel<T, true, false, true>));
      } else {
        gpuErrchk(cudaFuncGetAttributes(
            &funcAttrib,
            detail::computeGlobalHistogramKernel<T, false, false, true>));
      }
    }
    struct cudaDeviceProp& prop = utils::getDeviceProperties();

    auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
    auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
    size_t const hist_size = sizeof(size_t) * alphabet_size_;

    auto maxThreadsPerBlockHist = std::min(
        utils::kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
    maxThreadsPerBlockHist =
        utils::findLargestDivisor(utils::kMaxTPB, maxThreadsPerBlockHist);

    auto const hists_per_SM = max_shmem_per_SM / hist_size;

    auto min_block_size =
        hists_per_SM < utils::kMinBPM
            ? utils::kMinTPB
            : std::max(utils::kMinTPB, static_cast<uint32_t>(
                                           max_threads_per_SM / hists_per_SM));

    // Make the minimum block size a multiple of WS
    min_block_size = ((min_block_size + WS - 1) / WS) * WS;
    // Compute global_histogram and change text to min_alphabet
    size_t num_warps = std::min(
        (data_size + WS - 1) / WS,
        static_cast<size_t>(
            (max_threads_per_SM * prop.multiProcessorCount + WS - 1) / WS));

    auto const [num_blocks, threads_per_block] = utils::getLaunchConfig(
        num_warps, min_block_size, maxThreadsPerBlockHist);

    uint16_t const blocks_per_SM = max_threads_per_SM / threads_per_block;

    size_t const used_shmem =
        std::min(max_shmem_per_SM / blocks_per_SM, prop.sharedMemPerBlock);

    uint16_t const hists_per_block = std::min(
        static_cast<size_t>(threads_per_block), used_shmem / hist_size);

    if (is_pow_two) {
      if (is_min_alphabet_) {
        if (hists_per_block > 0) {
          detail::computeGlobalHistogramKernel<T, true, true, true>
              <<<num_blocks, threads_per_block, used_shmem>>>(
                  *this, d_data, data_size, d_histogram, d_alphabet,
                  alphabet_size_, hists_per_block);
        } else {
          detail::computeGlobalHistogramKernel<T, true, true, false>
              <<<num_blocks, threads_per_block>>>(
                  *this, d_data, data_size, d_histogram, d_alphabet,
                  alphabet_size_, hists_per_block);
        }
      } else {
        if (hists_per_block > 0) {
          detail::computeGlobalHistogramKernel<T, false, true, true>
              <<<num_blocks, threads_per_block, used_shmem>>>(
                  *this, d_data, data_size, d_histogram, d_alphabet,
                  alphabet_size_, hists_per_block);
        } else {
          detail::computeGlobalHistogramKernel<T, false, true, false>
              <<<num_blocks, threads_per_block>>>(
                  *this, d_data, data_size, d_histogram, d_alphabet,
                  alphabet_size_, hists_per_block);
        }
      }
    } else {
      if (is_min_alphabet_) {
        if (hists_per_block > 0) {
          detail::computeGlobalHistogramKernel<T, true, false, true>
              <<<num_blocks, threads_per_block, used_shmem>>>(
                  *this, d_data, data_size, d_histogram, d_alphabet,
                  alphabet_size_, hists_per_block);
        } else {
          detail::computeGlobalHistogramKernel<T, true, false, false>
              <<<num_blocks, threads_per_block>>>(
                  *this, d_data, data_size, d_histogram, d_alphabet,
                  alphabet_size_, hists_per_block);
        }
      } else {
        if (hists_per_block > 0) {
          detail::computeGlobalHistogramKernel<T, false, false, true>
              <<<num_blocks, threads_per_block, used_shmem>>>(
                  *this, d_data, data_size, d_histogram, d_alphabet,
                  alphabet_size_, hists_per_block);
        } else {
          detail::computeGlobalHistogramKernel<T, false, false, false>
              <<<num_blocks, threads_per_block>>>(
                  *this, d_data, data_size, d_histogram, d_alphabet,
                  alphabet_size_, hists_per_block);
        }
      }
    }
    kernelCheck();
  }

  /*!
   * \brief Fill the level of the wavelet tree with the data.
   * \param bit_array Bit array to fill.
   * \param d_data Data to fill the bit array with, on the device.
   * \param data_size Size of the data.
   * \param level Level of the wavelet tree to fill.
   */
  __host__ void fillLevel(BitArray& bit_array, T* const d_data,
                          size_t const data_size,
                          uint32_t const level) const noexcept {
    struct cudaFuncAttributes funcAttrib;
    gpuErrchk(
        cudaFuncGetAttributes(&funcAttrib, detail::fillLevelKernel<T, true>));
    uint32_t maxThreadsPerBlockFillLevel = std::min(
        utils::kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
    gpuErrchk(
        cudaFuncGetAttributes(&funcAttrib, detail::fillLevelKernel<T, false>));
    maxThreadsPerBlockFillLevel =
        std::min(maxThreadsPerBlockFillLevel,
                 static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

    maxThreadsPerBlockFillLevel =
        utils::findLargestDivisor(utils::kMaxTPB, maxThreadsPerBlockFillLevel);

    struct cudaDeviceProp& prop = utils::getDeviceProperties();
    if (level == 0) {
      gpuErrchk(cudaFuncSetAttribute(
          detail::fillLevelKernel<T, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          prop.sharedMemPerBlockOptin - funcAttrib.sharedSizeBytes));
    }

    auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
    auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
    auto shmem_per_thread = sizeof(uint32_t) * 8 * sizeof(T);

    // Pad shmem banks if memory suffices
    size_t padded_shmem = std::numeric_limits<size_t>::max();
    if constexpr (utils::kBankSizeBytes <= sizeof(T)) {
      // If the bank size is smaller than or equal to T, one element of
      // padding per thread is needed.
      padded_shmem = shmem_per_thread * max_threads_per_SM +
                     sizeof(T) * max_threads_per_SM;
    } else {
      padded_shmem =
          shmem_per_thread * max_threads_per_SM +
          shmem_per_thread * max_threads_per_SM / utils::kBanksPerLine;
    }
    bool const enough_shmem =
        (padded_shmem + funcAttrib.sharedSizeBytes) <= max_shmem_per_SM;
    if (enough_shmem) {
      shmem_per_thread = padded_shmem / max_threads_per_SM;
    }

    int num_blocks, threads_per_block;

    size_t const num_warps = std::min(
        (data_size + WS - 1) / WS,
        static_cast<size_t>(
            (max_threads_per_SM * prop.multiProcessorCount + WS - 1) / WS));

    std::tie(num_blocks, threads_per_block) = utils::getLaunchConfig(
        num_warps, utils::kMinTPB, maxThreadsPerBlockFillLevel);

    if (enough_shmem) {
      detail::fillLevelKernel<T, true>
          <<<num_blocks, threads_per_block,
             shmem_per_thread * threads_per_block>>>(
              bit_array, d_data, data_size, alphabet_start_bit_, level,
              num_blocks * threads_per_block, threads_per_block / WS);
    } else {
      detail::fillLevelKernel<T, false>
          <<<num_blocks, threads_per_block,
             sizeof(uint32_t) * (threads_per_block / WS)>>>(
              bit_array, d_data, data_size, alphabet_start_bit_, level,
              num_blocks * threads_per_block, threads_per_block / WS);
    }
    kernelCheck();
  }

  /*!
   * \brief Get the node information for the wavelet tree.
   * \param alphabet Alphabet of the wavelet tree.
   * \param codes Codes of the alphabet.
   * \return Vector of node information.
   */
  __host__ [[nodiscard]] static std::vector<detail::NodeInfo<T>> getNodeInfos(
      std::vector<T> const& alphabet, std::vector<Code> const& codes) noexcept {
    auto const alphabet_size = alphabet.size();
    auto const symbol_len =
        static_cast<uint8_t>(utils::ceilLog2(alphabet_size));

    std::vector<Code> alphabet_codes(alphabet_size);
    for (size_t i = 0; i < alphabet_size; ++i) {
      if (i < alphabet_size - codes.size()) {
        alphabet_codes[i] = {symbol_len, static_cast<T>(i)};
      } else {
        alphabet_codes[i] =
            codes[i - static_cast<T>(alphabet_size - codes.size())];
      }
    }

    std::vector<detail::NodeInfo<T>> node_starts;
    for (uint8_t l = 1; l < symbol_len; ++l) {
      // Find each position where the l-th MSB of the symbol is 0 and the
      // previous 1
      for (size_t i = 1; i < alphabet_size; ++i) {
        if (alphabet_codes[i].len_ > l and
            (utils::getBit(symbol_len - l - 1, alphabet_codes[i].code_) ==
             0) and
            (utils::getBit(symbol_len - l - 1, alphabet_codes[i - 1].code_) ==
             1)) {
          node_starts.push_back({alphabet[i], l});
        }
      }
    }
    return node_starts;
  }

  /*!
   * \brief Get an upper bound of size of the GPU memory needed for the
   * wavelet tree.
   * \param data_size Size of the data.
   * \param alphabet_size Size of the alphabet.
   * \param num_levels Number of levels in the wavelet tree.
   * \param num_codes Number of codes in the alphabet.
   * \return Upper bound of size of the GPU memory needed for the wavelet
   * tree.
   */
  __host__ [[nodiscard]] static size_t getNeededGPUMemory(
      size_t const data_size, size_t const alphabet_size,
      uint8_t const num_levels, T const num_codes) noexcept {
    size_t total_size = 0;
    total_size += num_codes * sizeof(Code);      // d_codes_
    total_size += (num_levels - 1) * sizeof(T);  // d_num_nodes_at_level_
    total_size += alphabet_size *
                  (sizeof(size_t) +
                   sizeof(detail::NodeInfo<T>));   // Upper bound for d_ranks_
                                                   // and d_node_starts
    total_size += alphabet_size * sizeof(size_t);  // d_counts_
    total_size += BitArray::getNeededGPUMemory(data_size, num_levels);
    total_size += RankSelect::getNeededGPUMemory(data_size, num_levels);

    return total_size;
  }

 private:
  size_t alphabet_size_;       /*!< Size of the alphabet*/
  uint8_t alphabet_start_bit_; /*!< Bit where the alphabet starts, 0 is the
                                  least significant bit*/
  Code* d_codes_ =
      nullptr; /*!< Array of codes for each symbol in the alphabet*/
  size_t* d_counts_ = nullptr; /*!< Array of counts for each symbol*/
  std::vector<size_t> counts_; /*!< Array of counts for each symbol*/
  uint8_t num_levels_;         /*!< Number of levels in the wavelet tree*/
  bool is_min_alphabet_; /*!< Flag to signal whether the alphabet is already
                            minimal*/
  T codes_start_;        /*!< Minimal alphabet symbol where the codes start*/
  bool is_copy_;         /*!< Flag to signal whether current object is a copy*/
  T* access_pinned_mem_pool_; /*!< Pinned memory pool for access queries*/
  size_t access_mem_pool_size_ = 10'000'000;
  size_t* rank_pinned_mem_pool_; /*!< Pinned memory pool for rank queries*/
  size_t rank_mem_pool_size_ = 10'000'000;
  size_t* select_pinned_mem_pool_; /*!< Pinned memory pool for select queries*/
  size_t select_mem_pool_size_ = 10'000'000;

  T num_nodes_until_last_level_; /*!< Number of nodes until the last level*/
  size_t* d_ranks_ = nullptr;
  T* d_num_nodes_at_level_ =
      nullptr;  /*!< Number of nodes at each level, without counting nodes that
                   start at 0*/
  T num_ranks_; /*!< Number of precomputed ranks in the wavelet tree*/
  uint32_t GPU_index_; /*!< Index of the GPU to use*/
};
}  // namespace ecl
