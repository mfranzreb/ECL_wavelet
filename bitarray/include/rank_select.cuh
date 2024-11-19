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
  BitArray const bit_array_; /*!< Bitarray the object wraps.*/

  /*!
   * \brief Constructor. Creates the auxiliary information for efficient rank
   * and select queries.
   * \param bit_array \c BitArray to be used for queries.
   */
  __host__ RankSelect(BitArray const&& bit_array) noexcept;

  /*!
   * \brief Copy constructor.
   * \param other \c RankSelect object to be copied.
   */
  __host__ RankSelect(RankSelect const& other) noexcept;

  __host__ ~RankSelect();

  /*!
   * \brief Computes rank of zeros.
   * \param array_index Index of the bit array to be used.
   * \param index Index the rank of zeros is computed for.
   * \param t_id Thread ID, has to start at 0.
   * \param num_threads Number of threads accessing the function.
   * \return Number of zeros (rank) before position \c index.
   */
  __device__ [[nodiscard]] size_t rank0(uint32_t const array_index,
                                        size_t index, int t_id,
                                        int num_threads) const;

  /*!
   * \brief Computes rank of ones.
   * \param array_index Index of the bit array to be used.
   * \param index Index the rank of ones is computed for.
   * \param t_id Thread ID, has to start at 0.
   * \param num_threads Number of threads accessing the function. Right now 32
   * is assumed
   * \return Numbers of ones (rank) before position \c index.
   */
  __device__ [[nodiscard]] size_t rank1(uint32_t const array_index,
                                        size_t index, int t_id,
                                        int num_threads) const;

  /*!
   * \brief Get position of i-th zero or one. Starting from 1.
   * \tparam Value 0 for zeros, 1 for ones.
   * * \param array_index Index of the bit array to be used.
   * \param i Rank of zero/one the position is searched for.
   * \param local_t_id Thread ID, has to start at 0.
   * \param num_threads Number of threads accessing the function. Right now 32
   * is assumed.
   * \return Position of the i-th zero/one.
   */
  template <int Value>
  __device__ [[nodiscard]] size_t select(uint32_t const array_index, size_t i,
                                         int const local_t_id,
                                         int const num_threads);

  /*!
   * \brief Get the number of L1 blocks for a bit array.
   * \param array_index Index of the bit array to be used.
   * \return Number of L1 blocks.
   */
  __device__ [[nodiscard]] size_t getNumL1Blocks(
      uint32_t const array_index) const;

  __host__ [[nodiscard]] size_t getNumL1BlocksHost(
      uint32_t const array_index) const;

  /*!
   * \brief Get the number of L2 blocks for a bit array.
   * \param array_index Index of the bit array to be used.
   * \return Number of L2 blocks.
   */
  __device__ [[nodiscard]] size_t getNumL2Blocks(
      uint32_t const array_index) const;

  __host__ [[nodiscard]] size_t getNumL2BlocksHost(
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
   * \return Pointer to L1 entry.
   */
  __device__ [[nodiscard]] RankSelectConfig::L1_TYPE getL1Entry(
      uint32_t const array_index, size_t const index) const;

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
   * \brief Get the total number of L2 blocks.
   * \return Total number of L2 blocks.
   */
  __host__ __device__ [[nodiscard]] size_t getTotalNumL2Blocks() const;

 private:
  /*!
   * \brief Helper function to share a variable between all threads in a warp.
   * \tparam T Type of the variable to be shared.
   * \param condition Condition to be met for sharing the variable. Only one
   * thread should fulfill it.
   * \param var Variable to be shared.
   */
  template <typename T>
  __device__ void shareVar(bool condition, size_t& var);

  /*!
   * \brief Get the position of the n-th 0 or 1 bit in a word.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param n Rank of the bit.
   * \param word Word the bit is in.
   * \return Position of the n-th bit.
   */
  template <int Value>
  __device__ [[nodiscard]] size_t getNBitPos(size_t n, uint32_t word);

  RankSelectConfig::L1_TYPE*
      d_l1_indices_; /*!< Device pointer to L1 indices for all arrays.*/
  RankSelectConfig::L2_TYPE*
      d_l2_indices_; /*!< Device pointer to L2 indices for all arrays.*/
  size_t*
      d_l1_offsets_; /*!< Offsets where each L1 index for a bit array starts.*/
  size_t*
      d_l2_offsets_; /*!< Offsets where each L2 index for a bit array starts.*/
  std::vector<size_t> l2_offsets_; /*!< Offsets where each L2 index for a bit
                                    array starts. Not accessible from device.*/

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
 * \brief Create the necessary structures for rank and select queries.
 * \param bit_array BitArray to be used for queries.
 * \return RankSelect object with the necessary structures.
 */
__host__ RankSelect createRankSelectStructures(BitArray&& bit_array);

/*!
 * \brief Fill L2 indices and prepare L1 indices for prefix sum.
 * \param rank_select RankSelect object to fill indices for.
 * \param array_index Index of the bit array to be used.
 * \param num_blocks Number of blocks the kernel is called with.
 * \param entries_per_warp Number of entries to be processed by a warp.
 */
__global__ void calculateL2EntriesKernel(RankSelect rank_select,
                                         uint32_t const array_index,
                                         uint32_t const entries_per_warp);

__global__ void calculateLastL1BlockKernel(RankSelect rank_select,
                                           uint32_t const array_index,
                                           uint32_t const entries_per_warp,
                                           uint8_t const num_last_l2_blocks);
}  // namespace ecl
