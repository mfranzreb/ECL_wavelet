#pragma once
#include <bit_array.cuh>
#include <cstdint>

namespace ecl {
/*!
 * \brief Static configuration for \c RankSelect.
 */
struct RankSelectConfig {
  // Bits covered by an L2-block.
  static constexpr size_t L2_BIT_SIZE = 2048;

  // Number of L2-blocks per L1-block
  static constexpr size_t NUM_L2_PER_L1 = 32;

  // Bits covered by an L1-block.
  static constexpr size_t L1_BIT_SIZE = NUM_L2_PER_L1 * L2_BIT_SIZE;

  // Number of 32-bit words covered by an L2-block.
  static constexpr size_t L2_WORD_SIZE = L2_BIT_SIZE / (sizeof(uint32_t) * 8);
  // Number of 32-bit words covered by an L1-block.
  static constexpr size_t L1_WORD_SIZE = L1_BIT_SIZE / (sizeof(uint32_t) * 8);

  using L1_TYPE = uint64_t;
  using L2_TYPE = uint16_t;
};  // struct RankSelectConfiguration

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
  __device__ [[nodiscard]] size_t rank0(uint32_t const array_index,
                                        size_t const index) {
    return index - rank1<NumThreads>(array_index, index);
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
  __device__ [[nodiscard]] size_t rank1(uint32_t const array_index,
                                        size_t const index) {
    assert(array_index < bit_array_.numArrays());
    assert(index <= bit_array_.size(array_index));
    if (index == 0) {
      return 0;
    }
    uint8_t t_id = 0;
    if constexpr (NumThreads > 1) {
      t_id = threadIdx.x % NumThreads;
    }
    size_t const l1_pos = index / RankSelectConfig::L1_BIT_SIZE;
    size_t const l2_pos =
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

    // TODO: do popc outside of the if
    for (size_t i = start_word + t_id; i <= end_word; i += NumThreads) {
      // TODO: read word before if
      if (i == end_word) {
        // Only consider bits up to the index.
        result += __popc(bit_array_.partialWord(
            array_index, i, index % (sizeof(uint32_t) * 8)));
      } else {
        result += __popc(bit_array_.word(array_index, i));
      }
    }

    if constexpr (NumThreads > 1) {
      uint32_t mask = ~0;
      if constexpr (NumThreads < 32) {
        mask = ((1 << NumThreads) - 1)
               << (NumThreads * ((threadIdx.x % WS) / NumThreads));
      }
      result = warpSum<size_t, NumThreads>(mask, result);
      // communicate the result to all threads
      shareVar<size_t>(t_id == 0, result, mask);
    }

    return result;
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
                                       uint8_t const value) noexcept;

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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    //? do all threads get the result?
    val = __reduce_add_sync(mask, val);
#else
    for (int offset = NumThreads / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
    }
    return val;
#endif
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
    __syncwarp(mask);
    uint32_t src_thread = __ballot_sync(mask, condition);
    // Get the value from the first thread that fulfills the condition
    src_thread = __ffs(src_thread) - 1;
    __syncwarp(mask);
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

  uint8_t* d_num_last_l2_blocks_; /*!< Number of L2 blocks in the last L1 block
                                   for each bit array.*/
  size_t* d_num_l1_blocks_;       /*!< Number of L1 blocks for each bit array.*/
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
__global__ void calculateL2EntriesKernel(RankSelect rank_select,
                                         uint32_t const array_index,
                                         uint8_t const num_last_l2_blocks);

}  // namespace ecl
