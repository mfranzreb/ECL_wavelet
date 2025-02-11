#pragma once
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <bit_array.cuh>
#include <cstdint>
#include <cub/block/block_scan.cuh>

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
  static constexpr size_t SELECT_SAMPLE_RATE = 4096;

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
  template <uint32_t Value, int NumThreads>
  __device__ [[nodiscard]] size_t select(
      uint32_t const array_index, size_t i, size_t const BA_offset,
      cub::WarpScan<RSConfig::L2_TYPE, NumThreads>::TempStorage* temp_storage) {
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

    size_t const prev_sample_pos = i / RSConfig::SELECT_SAMPLE_RATE;
    size_t result = 1;  // 1 so that "curr_l2_block" is not 0 if no sample found
    if (prev_sample_pos > 0) {
      if constexpr (Value == 0) {
        result = d_select_samples_0_[d_select_samples_0_offsets_[array_index] +
                                     prev_sample_pos - 1];
      } else {
        result = d_select_samples_1_[d_select_samples_1_offsets_[array_index] +
                                     prev_sample_pos - 1];
      }
    }

    // TODO: reduce register usage here
    //  Find next L1 block where the i-th zero/one is, starting from result
    size_t const l1_offset = d_l1_offsets_[array_index];
    size_t const l2_offset = d_l2_offsets_[array_index];
    RSConfig::L2_TYPE const num_last_l2_blocks =
        d_num_last_l2_blocks_[array_index];
    size_t const num_l1_blocks = d_num_l1_blocks_[array_index];

    // Starting from 1 since 0-th entry always 0.
    size_t curr_l1_block = local_t_id + (result + RSConfig::L1_BIT_SIZE - 1) /
                                            RSConfig::L1_BIT_SIZE;
    uint32_t active_threads;
    if constexpr (NumThreads > 1) {
      active_threads = __ballot_sync(mask, curr_l1_block < num_l1_blocks);
    }
    size_t current_val = 0;
    size_t has_i_before = 0;
    while (curr_l1_block < num_l1_blocks) {
      if constexpr (Value == 0) {
        current_val = curr_l1_block * RSConfig::L1_BIT_SIZE -
                      d_l1_indices_[l1_offset + curr_l1_block];
      } else {
        current_val = d_l1_indices_[l1_offset + curr_l1_block];
      }
      has_i_before = current_val >= i ? curr_l1_block : 0;
      if constexpr (NumThreads > 1) {
        shareVar<size_t>(current_val >= i, has_i_before, active_threads);
      }
      if (has_i_before != 0) {
        break;
      }
      curr_l1_block += NumThreads;
      if constexpr (NumThreads > 1) {
        active_threads =
            __ballot_sync(active_threads, curr_l1_block < num_l1_blocks);
      }
    }
    if constexpr (NumThreads > 1) {
      shareVar<size_t>(current_val >= i, has_i_before, mask);
    }
    if (has_i_before != 0) {
      curr_l1_block = has_i_before - 1;
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
    RSConfig::L2_TYPE const l1_block_length =
        has_i_before == 0 ? num_last_l2_blocks : RSConfig::NUM_L2_PER_L1;
    size_t const l1_block_start =
        (result / RSConfig::L1_BIT_SIZE) * RSConfig::NUM_L2_PER_L1;
    RSConfig::L2_TYPE const blocks_per_thread =
        (l1_block_length + NumThreads - 1) / NumThreads;

    size_t const start_l2_block =
        l1_block_start + local_t_id * blocks_per_thread;
    size_t const end_l2_block = min(l1_block_start + l1_block_length,
                                    start_l2_block + blocks_per_thread);

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
        has_i_before == 0 ? bit_array_.sizeInWords(array_index) -
                                (result / (sizeof(uint32_t) * 8))
                          : RSConfig::L2_WORD_SIZE;

    size_t const word_start = result / (sizeof(uint32_t) * 8);
    result = 0;  // 0 signalizes that nothing was found
    if constexpr (NumThreads > 1) {
      cub::WarpScan<RSConfig::L2_TYPE, NumThreads> warp_scan(*temp_storage);
      for (size_t current_group_word = word_start;
           current_group_word < word_start + l2_block_length;
           current_group_word += NumThreads) {
        size_t const local_word = current_group_word + local_t_id;
        uint32_t word;
        if constexpr (Value == 0) {
          word = ~0;
        } else {
          word = 0;
        }
        RSConfig::L2_TYPE num_vals = 0;
        if (local_word < word_start + l2_block_length) {
          word = bit_array_.word(array_index, local_word, BA_offset);
          if constexpr (Value == 0) {
            num_vals = (sizeof(uint32_t) * 8) - __popc(word);
          } else {
            num_vals = __popc(word);
          }
        }
        RSConfig::L2_TYPE cum_vals = 0;

        // inclusive prefix sum
        warp_scan.InclusiveSum(num_vals, cum_vals);

        // Check if the i-th zero/one is in the current word
        RSConfig::L2_TYPE const vals_at_start = cum_vals - num_vals;
        if (cum_vals >= i and vals_at_start < i) {
          // Find the position of the i-th zero/one in the word
          // 1-indexed to distinguish from having found nothing, which is
          // 0.
          // TODO: faster implementations possible
          i -= vals_at_start;
          result = local_word * (sizeof(uint32_t) * 8) +
                   getNBitPos<Value>(i, word) + 1;
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
      for (size_t j = 0; j < l2_block_length; j++) {
        uint32_t word;
        if constexpr (Value == 0) {
          word = ~0;
        } else {
          word = 0;
        }
        RSConfig::L2_TYPE current_num_vals = 0;

        word = bit_array_.word(array_index, word_start + j, BA_offset);
        if constexpr (Value == 0) {
          current_num_vals = (sizeof(uint32_t) * 8) - __popc(word);
        } else {
          current_num_vals = __popc(word);
        }
        if ((num_vals + current_num_vals) >= i and num_vals < i) {
          i -= num_vals;
          result = (word_start + j) * (sizeof(uint32_t) * 8) +
                   getNBitPos<Value>(i, word) + 1;
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

  /*!
   * \brief Get the position of the n-th 0 or 1 bit in a word, starting from the
   * least significant bit.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param n Rank of the bit. Starts from 1.
   * \param word Word the bit is in.
   * \return Position of the n-th bit. Starts from 0.
   */
  template <uint32_t Value>
  __device__ [[nodiscard]] uint8_t getNBitPos(uint8_t const n, uint32_t word) {
    static_assert(Value == 0 or Value == 1,
                  "Template parameter must be 0 or 1");
    assert(n > 0);
    assert(n <= (sizeof(uint32_t) * 8));
    if constexpr (Value == 0) {
      // Find the position of the n-th zero in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word | (word + 1);  // set least significant 0-bit
      }
      return __ffs(~word) - 1;

    } else {
      // Find the position of the n-th one in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word & (word - 1);  // clear least significant 1-bit
      }
      return __ffs(word) - 1;
    }
  }

 private:
  // TODO: num_l1_blocks and l2 blocks not necessary
  RSConfig::L1_TYPE*
      d_l1_indices_; /*!< Device pointer to L1 indices for all arrays.*/
  RSConfig::L2_TYPE*
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

  size_t* d_select_samples_0_; /*!< Sampled positions for select queries.*/
  size_t* d_select_samples_0_offsets_; /*!< Offsets where each array's select
                                    samples start.*/
  size_t* d_select_samples_1_; /*!< Sampled positions for select queries.*/
  size_t* d_select_samples_1_offsets_; /*!< Offsets where each array's select
                                    samples start.*/
  size_t* d_total_num_ones_; /*!< Total number of ones for each array.*/

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
// TODO: fuse bothe kernels together
template <int ItemsPerThread>
__global__ LB(MAX_TPB, MIN_BPM) void calculateL2EntriesKernel(
    RankSelect rank_select, uint32_t const array_index,
    uint16_t const num_last_l2_blocks) {
  assert(blockDim.x % WS == 0);
  static_assert(ItemsPerThread > 0, "ItemsPerThread must be greater than 0.");
  int constexpr kNeededThreads = RSConfig::NUM_L2_PER_L1 / ItemsPerThread;
  __shared__ RSConfig::L2_TYPE l2_entries[RSConfig::NUM_L2_PER_L1];
  __shared__ RSConfig::L1_TYPE next_l1_entry;

  __shared__
      typename cub::BlockScan<RSConfig::L2_TYPE, kNeededThreads>::TempStorage
          scan_storage;

  uint32_t const warp_id = threadIdx.x / WS;
  uint32_t const local_t_id = threadIdx.x % WS;
  uint32_t const num_warps = blockDim.x / WS;
  if (blockIdx.x < gridDim.x - 1) {
    size_t const offset = rank_select.bit_array_.getOffset(array_index);
    // find L1 block index
    uint32_t const l1_index = blockIdx.x;

    for (uint32_t i = warp_id; i < RSConfig::NUM_L2_PER_L1; i += num_warps) {
      RSConfig::L2_TYPE num_ones = 0;
      size_t const start_word =
          l1_index * RSConfig::L1_WORD_SIZE + i * RSConfig::L2_WORD_SIZE;

      size_t const end_word = start_word + RSConfig::L2_WORD_SIZE;
      for (size_t j = start_word + 2 * local_t_id; j < end_word; j += 2 * WS) {
        // Global memory load
        // load as 64 bits.
        uint64_t const word =
            rank_select.bit_array_.twoWords(array_index, j, offset);
        num_ones += __popcll(word);
      }

      // Warp reduction
      RSConfig::L2_TYPE const total_ones =
          rank_select.warpReduce<RSConfig::L2_TYPE, WS>(~0, num_ones);
      __syncwarp();

      if (local_t_id == 0) {
        l2_entries[i] = total_ones;
      }
    }

    __syncthreads();
    // perform block exclusive sum of l2 entries
    if (threadIdx.x < kNeededThreads) {
      RSConfig::L2_TYPE l2_entries_t[ItemsPerThread];
      for (uint32_t i = 0; i < ItemsPerThread; i++) {
        l2_entries_t[i] = l2_entries[threadIdx.x * ItemsPerThread + i];
      }

      // last thread writes it's entry to the following L1 block
      if (threadIdx.x == kNeededThreads - 1) {
        next_l1_entry = l2_entries_t[ItemsPerThread - 1];
      }
      cub::BlockScan<RSConfig::L2_TYPE, kNeededThreads>(scan_storage)
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
                                 l1_index * RSConfig::NUM_L2_PER_L1 +
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

    auto const l1_start_word =
        (rank_select.getNumL1Blocks(array_index) - 1) * RSConfig::L1_WORD_SIZE;

    for (uint32_t i = warp_id; i < num_last_l2_blocks; i += num_warps) {
      RSConfig::L2_TYPE num_ones = 0;
      size_t const start_word = l1_start_word + i * RSConfig::L2_WORD_SIZE;

      size_t const end_word =
          min(rank_select.bit_array_.sizeInWords(array_index),
              start_word + RSConfig::L2_WORD_SIZE);
      for (size_t j = start_word + local_t_id; j < end_word; j += WS) {
        uint32_t const word = rank_select.bit_array_.word(array_index, j);

        // Compute num ones even of last word, since it will not be saved.
        num_ones += __popc(word);
      }

      // Warp reduction
      RSConfig::L2_TYPE const total_ones =
          rank_select.warpReduce<RSConfig::L2_TYPE, WS>(~0, num_ones);
      __syncwarp();
      if (local_t_id == 0) {
        l2_entries[i] = total_ones;
      }
    }

    __syncthreads();
    // perform block exclusive sum of l2 entries
    if (threadIdx.x < kNeededThreads) {
      RSConfig::L2_TYPE l2_entries_t[ItemsPerThread];
      for (uint32_t i = 0; i < ItemsPerThread; i++) {
        if (threadIdx.x * ItemsPerThread + i < num_last_l2_blocks) {
          l2_entries_t[i] = l2_entries[threadIdx.x * ItemsPerThread + i];
        } else {
          l2_entries_t[i] = 0;
        }
      }

      cub::BlockScan<RSConfig::L2_TYPE, kNeededThreads>(scan_storage)
          .ExclusiveSum(l2_entries_t, l2_entries_t);

      // All threads write their result to global memory
      for (uint32_t i = 0; i < ItemsPerThread; i++) {
        if (threadIdx.x * ItemsPerThread + i < num_last_l2_blocks) {
          rank_select.writeL2Index(array_index,
                                   (rank_select.getNumL1Blocks(array_index) -
                                    1) * RSConfig::NUM_L2_PER_L1 +
                                       threadIdx.x * ItemsPerThread + i,
                                   l2_entries_t[i]);
        }
      }
    }
  }
  return;
}

template <int ThreadsPerIndex>
__global__ LB(MAX_TPB, MIN_BPM) void calculateSelectSamplesKernel(
    RankSelect rank_select, uint32_t const array_index, size_t const num_groups,
    size_t const total_num_l2_blocks, size_t const bit_array_size_in_words,
    size_t* total_num_ones) {
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t group_counter[];
  __shared__ typename cub::WarpScan<size_t,
                                    ThreadsPerIndex>::TempStorage
      temp_storage[1024 / ThreadsPerIndex];  // TODO

  uint8_t constexpr start_l2_block =
      (RSConfig::SELECT_SAMPLE_RATE / RSConfig::L2_BIT_SIZE) - 1;

  size_t const global_group_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / ThreadsPerIndex;
  uint16_t const local_group_id = threadIdx.x / ThreadsPerIndex;
  uint8_t const local_t_id = threadIdx.x % ThreadsPerIndex;

  for (size_t curr_l2_block = global_group_id + start_l2_block;
       curr_l2_block < total_num_l2_blocks; curr_l2_block += num_groups) {
    if (local_t_id == 0) {
      group_counter[local_group_id] = rank_select.getL1Entry(
          array_index, curr_l2_block / RSConfig::NUM_L2_PER_L1);
      group_counter[local_group_id] +=
          rank_select.getL2Entry(array_index, curr_l2_block);
    }
    size_t const start_word = curr_l2_block * RSConfig::L2_WORD_SIZE;
    size_t const end_word =
        min(start_word + RSConfig::L2_WORD_SIZE, bit_array_size_in_words);
    cub::WarpScan<size_t, ThreadsPerIndex> warp_scan(
        temp_storage[local_group_id]);
    for (size_t i = start_word; i < end_word; i += ThreadsPerIndex) {
      size_t local_i = i + local_t_id;
      uint32_t word = 0;
      bool const is_last_word = local_i == bit_array_size_in_words - 1;
      uint8_t const word_size =
          is_last_word ? (rank_select.bit_array_.size(array_index) - 1) %
                                 (sizeof(uint32_t) * 8) +
                             1
                       : sizeof(uint32_t) * 8;
      if (local_i < end_word) {
        word = rank_select.bit_array_.word(array_index, local_i);
        if (is_last_word) {
          word = rank_select.bit_array_.partialWord(word, word_size);
        }
      }
      size_t const num_ones = __popc(word);
      size_t cum_ones = 0;
      warp_scan.InclusiveSum(num_ones, cum_ones);
      cum_ones += group_counter[local_group_id];
      size_t const ones_at_start = cum_ones - num_ones;
      if (cum_ones / RSConfig::SELECT_SAMPLE_RATE >
              ones_at_start / RSConfig::SELECT_SAMPLE_RATE and
          local_i < end_word) {
        size_t const sample_pos =
            local_i * (sizeof(uint32_t) * 8) +
            rank_select.getNBitPos<1>(
                (RSConfig::SELECT_SAMPLE_RATE -
                 ones_at_start % RSConfig::SELECT_SAMPLE_RATE),
                word);
        rank_select.writeSelectSample<1>(
            array_index, ones_at_start / RSConfig::SELECT_SAMPLE_RATE,
            sample_pos);
      }
      size_t const zeros_at_start =
          local_i * (sizeof(uint32_t) * 8) - ones_at_start;
      size_t const cum_zeros = zeros_at_start + word_size - num_ones;

      if (cum_zeros / RSConfig::SELECT_SAMPLE_RATE >
              zeros_at_start / RSConfig::SELECT_SAMPLE_RATE and
          local_i < end_word) {
        size_t const sample_pos =
            local_i * (sizeof(uint32_t) * 8) +
            rank_select.getNBitPos<0>(
                (RSConfig::SELECT_SAMPLE_RATE -
                 zeros_at_start % RSConfig::SELECT_SAMPLE_RATE),
                word);
        rank_select.writeSelectSample<0>(
            array_index, zeros_at_start / RSConfig::SELECT_SAMPLE_RATE,
            sample_pos);
      }
      if (local_t_id == ThreadsPerIndex - 1) {
        group_counter[local_group_id] = cum_ones;
      }
      if (is_last_word and local_i < end_word) {
        assert(rank_select.bit_array_.size(array_index) ==
               cum_ones + cum_zeros);
        total_num_ones[0] = cum_ones;
      }
    }
  }
  if (total_num_l2_blocks <= start_l2_block and global_group_id == 0) {
    // Get the number of ones in the last L2 block
    if (local_t_id == 0) {
      group_counter[local_group_id] =
          rank_select.getL2Entry(array_index, total_num_l2_blocks - 1);
    }
    size_t const start_word =
        (total_num_l2_blocks - 1) * RSConfig::L2_WORD_SIZE;
    size_t const end_word =
        min(start_word + RSConfig::L2_WORD_SIZE, bit_array_size_in_words);
    for (size_t i = start_word; i < end_word; i += ThreadsPerIndex) {
      size_t local_i = i + local_t_id;
      uint32_t word = 0;
      bool const is_last_word = local_i == bit_array_size_in_words - 1;
      uint8_t const word_size =
          is_last_word ? (rank_select.bit_array_.size(array_index) - 1) %
                                 (sizeof(uint32_t) * 8) +
                             1
                       : sizeof(uint32_t) * 8;
      if (local_i < end_word) {
        word = rank_select.bit_array_.word(array_index, local_i);
        if (is_last_word) {
          word = rank_select.bit_array_.partialWord(word, word_size);
        }
      }
      size_t num_ones = __popc(word);
      num_ones = rank_select.warpReduce<size_t, WS>(~0U, num_ones);
      shareVar<size_t>(local_t_id == 0, num_ones, ~0U);
      if (local_t_id == 0) {
        group_counter[local_group_id] += num_ones;
      }
      if (is_last_word) {
        total_num_ones[0] = group_counter[local_group_id];
      }
    }
  }
}
}  // namespace ecl
