#pragma once
#include <bit_array.cuh>
#include <cstdint>
#include <cub/block/block_scan.cuh>

namespace ecl {
/*!
 * \brief Static configuration for \c RankSelect.
 */
struct RankSelectConfig {
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

  using L1_TYPE = uint64_t;
  using L2_TYPE = uint16_t;
};  // struct RankSelectConfiguration

struct RankResult {
  size_t rank;
  bool bit;
};

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
  __host__ RankSelect(BitArray&& bit_array, uint8_t const GPU_index) noexcept;

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
  template <int NumThreads>
  __device__ [[nodiscard]] RankResult rank0(uint32_t const array_index,
                                            size_t const index,
                                            size_t const offset) {
    auto result = rank1<NumThreads>(array_index, index, offset);
    result.rank = index - result.rank;
    return result;
  }

  /*!
   * \brief Computes rank of ones.
   * \tparam NumThreads Number of threads accessing the function.
   * \param array_index Index of the bit array to be used.
   * \param index Index the rank of zeros is computed for.
   * \return Numbers of ones (rank) before position \c index, i.e. in the slice
   * [0, i).
   */
  template <int NumThreads>
  __device__ [[nodiscard]] RankResult rank1(uint8_t const array_index,
                                            size_t const index,
                                            size_t const offset) {
    assert(array_index < bit_array_.numArrays());
    assert(index <= bit_array_.size(array_index));
    if (index == 0) {
      return RankResult{0, bit_array_.access(array_index, 0, offset)};
    }
    uint8_t t_id = 0;
    if constexpr (NumThreads > 1) {
      t_id = threadIdx.x % NumThreads;
    }
    size_t const l1_pos = index / RankSelectConfig::L1_BIT_SIZE;
    uint16_t const l2_pos =
        (index % RankSelectConfig::L1_BIT_SIZE) / RankSelectConfig::L2_BIT_SIZE;
    // Only first thread in the local block stores the result
    size_t result =
        (d_l1_indices_[d_l1_offsets_[array_index] + l1_pos] +
         d_l2_indices_[d_l2_offsets_[array_index] +
                       l1_pos * RankSelectConfig::NUM_L2_PER_L1 + l2_pos]) *
        (t_id == 0);
    size_t const start_word = l1_pos * RankSelectConfig::L1_WORD_SIZE +
                              l2_pos * RankSelectConfig::L2_WORD_SIZE;
    size_t const end_word = index / (sizeof(uint32_t) * 8);

    uint64_t word;
    uint8_t bit_at_index;
    bool has_last_word = false;
    uint8_t const bit_index = index % (sizeof(uint64_t) * 8);
    for (size_t i = start_word + 2 * t_id; i <= end_word; i += 2 * NumThreads) {
      word = bit_array_.twoWords(array_index, i, offset);
      if (end_word == 0 or i >= end_word - 1) {
        // Only consider bits up to the index.
        bit_at_index = static_cast<uint8_t>((word >> bit_index) & 1U);
        word = bit_array_.partialTwoWords(word, bit_index);
        has_last_word = true;
      }
      result += __popcll(word);
    }

    if constexpr (NumThreads > 1) {
      uint32_t mask = ~0;
      if constexpr (NumThreads < WS) {
        mask = ((1 << NumThreads) - 1)
               << (NumThreads * ((threadIdx.x % WS) / NumThreads));
      }
      result = warpSum<size_t, NumThreads>(mask, result);
      // communicate the result to all threads
      shareVar<size_t>(t_id == 0, result, mask);

      shareVar<uint8_t>(has_last_word, bit_at_index, mask);
    }

    return RankResult{result, static_cast<bool>(bit_at_index)};
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
  template <uint32_t Value>
  __device__ [[nodiscard]] size_t select(uint32_t const array_index, size_t i,
                                         uint32_t const local_t_id,
                                         uint32_t const num_threads);

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

  /*!
   * \brief Write a value to the L2 index.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L2 index to be written to.
   * \param value Value to be written.
   */
  __device__ void writeL2Index(uint32_t const array_index, size_t const index,
                               RankSelectConfig::L2_TYPE const value) noexcept;

  /*!
   * \brief Write a value to the L1 index.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L1 index to be written to.
   * \param value Value to be written.
   */
  __device__ void writeL1Index(uint32_t const array_index, size_t const index,
                               RankSelectConfig::L1_TYPE const value) noexcept;

  /*!
   * \brief Get an L1 entry for a bit array.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L1 entry to be returned.
   * \return L1 entry.
   */
  __device__ [[nodiscard]] RankSelectConfig::L1_TYPE getL1Entry(
      uint32_t const array_index, size_t const index) const;

  /*!
   * \brief Get a device pointer to an L1 entry for a bit array.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L1 entry to be returned.
   * \return Pointer to L1 entry.
   */
  __host__ [[nodiscard]] RankSelectConfig::L1_TYPE* getL1EntryPointer(
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
  __device__ T warpSum(uint32_t const mask, T val) {
#pragma unroll
    for (int offset = NumThreads / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
    }
    return val;
  }

 private:
  /*!
   * \brief Helper function to share a variable between all threads in a warp.
   * \tparam T Type of the variable to be shared. Must be an integral or
   * floating point type.
   * \param condition Condition to be met for sharing the
   * variable. Only one thread should fulfill it.
   * \param var Variable to be shared.
   * \param mask Mask representing the threads that should share the variable.
   */
  template <typename T>
  __device__ void shareVar(bool condition, T& var, uint32_t const mask) {
    static_assert(
        std::is_integral<T>::value or std::is_floating_point<T>::value,
        "T must be an integral or floating-point type.");
    uint32_t src_thread = __ballot_sync(mask, condition);
    // Get the value from the first thread that fulfills the condition
    src_thread = __ffs(src_thread) - 1;
    var = __shfl_sync(mask, var, src_thread);
  }

  /*!
   * \brief Get the position of the n-th 0 or 1 bit in a word, starting from the
   * least significant bit.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param n Rank of the bit. Starts from 1.
   * \param word Word the bit is in.
   * \return Position of the n-th bit. Starts from 0.
   */
  template <uint32_t Value>
  __device__ [[nodiscard]] uint8_t getNBitPos(uint8_t const n, uint32_t word);

  RankSelectConfig::L1_TYPE*
      d_l1_indices_; /*!< Device pointer to L1 indices for all arrays.*/
  RankSelectConfig::L2_TYPE*
      d_l2_indices_; /*!< Device pointer to L2 indices for all arrays.*/

  size_t*
      d_l1_offsets_; /*!< Offsets where each L1 index for a bit array starts.*/
  size_t*
      d_l2_offsets_; /*!< Offsets where each L2 index for a bit array starts.*/

  uint16_t* d_num_last_l2_blocks_; /*!< Number of L2 blocks in the last L1 block
                                   for each bit array.*/
  size_t* d_num_l1_blocks_; /*!< Number of L1 blocks for each bit array.*/
  std::vector<size_t> num_l1_blocks_; /*!< Number of L1 blocks for all bit
                                      arrays. Not accessible from device.*/
  size_t total_num_l2_blocks_;        /*!< Total number of L2 blocks for all bit
                                         arrays.*/
  bool is_copy_; /*!< Flag to signal whether current object is a
                    copy.*/
};  // class RankSelect

/*!
 * \brief Fill L2 indices and prepare L1 indices for prefix sum.
 * \param rank_select RankSelect object to fill indices for.
 * \param array_index Index of the bit array to be used.
 * \param num_last_l2_blocks Number of L2 blocks in the last L1 block.
 */
// TODO: reoptimize and retune kernel
template <int ItemsPerThread>
__global__ LB(MAX_TPB, MIN_BPM) void calculateL2EntriesKernel(
    RankSelect rank_select, uint32_t const array_index,
    uint16_t const num_last_l2_blocks) {
  assert(blockDim.x % WS == 0);
  static_assert(ItemsPerThread > 0, "ItemsPerThread must be greater than 0.");
  int constexpr kNeededThreads =
      RankSelectConfig::NUM_L2_PER_L1 / ItemsPerThread;
  __shared__ RankSelectConfig::L2_TYPE
      l2_entries[RankSelectConfig::NUM_L2_PER_L1];
  __shared__ RankSelectConfig::L1_TYPE next_l1_entry;

  __shared__ typename cub::BlockScan<RankSelectConfig::L2_TYPE,
                                     kNeededThreads>::TempStorage scan_storage;

  uint32_t const warp_id = threadIdx.x / WS;
  uint32_t const local_t_id = threadIdx.x % WS;
  uint32_t const num_warps = blockDim.x / WS;
  if (blockIdx.x < gridDim.x - 1) {
    size_t const offset = rank_select.bit_array_.getOffset(array_index);
    // find L1 block index
    uint32_t const l1_index = blockIdx.x;

    for (uint32_t i = warp_id; i < RankSelectConfig::NUM_L2_PER_L1;
         i += num_warps) {
      RankSelectConfig::L2_TYPE num_ones = 0;
      size_t const start_word = l1_index * RankSelectConfig::L1_WORD_SIZE +
                                i * RankSelectConfig::L2_WORD_SIZE;

      size_t const end_word = start_word + RankSelectConfig::L2_WORD_SIZE;
      for (size_t j = start_word + 2 * local_t_id; j < end_word; j += 2 * WS) {
        // Global memory load
        // load as 64 bits.
        uint64_t const word =
            rank_select.bit_array_.twoWords(array_index, j, offset);
        num_ones += __popcll(word);
      }

      // Warp reduction
      RankSelectConfig::L2_TYPE const total_ones =
          rank_select.warpSum<RankSelectConfig::L2_TYPE, WS>(~0, num_ones);
      __syncwarp();

      if (local_t_id == 0) {
        l2_entries[i] = total_ones;
      }
    }

    __syncthreads();
    // perform block exclusive sum of l2 entries
    if (threadIdx.x < kNeededThreads) {
      RankSelectConfig::L2_TYPE l2_entries_t[ItemsPerThread];
      for (uint32_t i = 0; i < ItemsPerThread; i++) {
        l2_entries_t[i] = l2_entries[threadIdx.x * ItemsPerThread + i];
      }

      // last thread writes it's entry to the following L1 block
      if (threadIdx.x == kNeededThreads - 1) {
        next_l1_entry = l2_entries_t[ItemsPerThread - 1];
      }
      cub::BlockScan<RankSelectConfig::L2_TYPE, kNeededThreads>(scan_storage)
          .ExclusiveSum(l2_entries_t, l2_entries_t);

      // last thread adds it's entry to the following L1 block
      if (threadIdx.x == kNeededThreads - 1) {
        next_l1_entry += l2_entries_t[ItemsPerThread - 1];
        rank_select.writeL1Index(array_index, l1_index + 1, next_l1_entry);
      }
      // All threads write their result to global memory
      // Possibly uncoalesced global memory store
      for (uint32_t i = 0; i < ItemsPerThread; i++) {
        rank_select.writeL2Index(array_index,
                                 l1_index * RankSelectConfig::NUM_L2_PER_L1 +
                                     threadIdx.x * ItemsPerThread + i,
                                 l2_entries_t[i]);
      }
    }
  }

  else {
    if (threadIdx.x == 0) {
      rank_select.writeNumLastL2Blocks(array_index, num_last_l2_blocks);
    }
    if (num_last_l2_blocks == 1) {
      return;
    }

    auto const l1_start_word = (rank_select.getNumL1Blocks(array_index) - 1) *
                               RankSelectConfig::L1_WORD_SIZE;

    for (uint32_t i = warp_id; i < num_last_l2_blocks; i += num_warps) {
      RankSelectConfig::L2_TYPE num_ones = 0;
      size_t const start_word =
          l1_start_word + i * RankSelectConfig::L2_WORD_SIZE;

      size_t const end_word =
          min(rank_select.bit_array_.sizeInWords(array_index),
              start_word + RankSelectConfig::L2_WORD_SIZE);
      for (size_t j = start_word + local_t_id; j < end_word; j += WS) {
        uint32_t const word = rank_select.bit_array_.word(array_index, j);

        // Compute num ones even of last word, since it will not be saved.
        num_ones += __popc(word);
      }

      // Warp reduction
      RankSelectConfig::L2_TYPE const total_ones =
          rank_select.warpSum<RankSelectConfig::L2_TYPE, WS>(~0, num_ones);
      __syncwarp();
      if (local_t_id == 0) {
        l2_entries[i] = total_ones;
      }
    }

    __syncthreads();
    // perform block exclusive sum of l2 entries
    if (threadIdx.x < kNeededThreads) {
      RankSelectConfig::L2_TYPE l2_entries_t[ItemsPerThread];
      for (uint32_t i = 0; i < ItemsPerThread; i++) {
        if (threadIdx.x * ItemsPerThread + i < num_last_l2_blocks) {
          l2_entries_t[i] = l2_entries[threadIdx.x * ItemsPerThread + i];
        } else {
          l2_entries_t[i] = 0;
        }
      }

      cub::BlockScan<RankSelectConfig::L2_TYPE, kNeededThreads>(scan_storage)
          .ExclusiveSum(l2_entries_t, l2_entries_t);

      // All threads write their result to global memory
      for (uint32_t i = 0; i < ItemsPerThread; i++) {
        if (threadIdx.x * ItemsPerThread + i < num_last_l2_blocks) {
          rank_select.writeL2Index(array_index,
                                   (rank_select.getNumL1Blocks(array_index) -
                                    1) * RankSelectConfig::NUM_L2_PER_L1 +
                                       threadIdx.x * ItemsPerThread + i,
                                   l2_entries_t[i]);
        }
      }
    }
  }
  return;
}

}  // namespace ecl
