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
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <cstdint>
#include <cub/device/device_scan.cuh>

#include "ecl_wavelet/bitarray/bit_array.cuh"

namespace ecl {
/*!
 * \brief Compile-time configuration for \c RankSelect.
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
};  // struct RSConfig

/*!
 * \brief Return struct of a binary rank operation.alignas
 * \details \c rank refers to the resulting rank, and \c bit to the bit at the
 * position that was given as input.
 */
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
  RankSelect() noexcept = default;

  /*!
   * \brief Constructor. Creates the auxiliary information for efficient rank
   * and select queries.
   * \param bit_array \c BitArray to be used for queries.
   * \param GPU_index Index of the GPU to be used.
   */
  __host__ RankSelect(BitArray&& bit_array, uint32_t const GPU_index) noexcept;

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

  /*!
   * \brief Destructor.
   */
  __host__ ~RankSelect();

  /*!
   * \brief Computes rank of zeros or ones for a bit array.
   * \tparam NumThreads Number of threads accessing the function. Must be less
   * than or equal to 32.
   * \tparam GetBit If true, the bit at the index is also returned.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param array_index Index of the bit array to be used.
   * \param index Index the rank of \c Value is computed for.
   * \param offset Index in the underlying array where the bit array starts.
   * \return Number of \c Value occurrences (rank) before position \c index,
   * i.e. in the slice [0, i).
   */
  // TODO: try to make loop coalesced
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
    } else if (index == bit_array_.size(array_index)) {
      if constexpr (Value == 0) {
        if constexpr (GetBit) {
          return RankResult{index - d_total_num_ones_[array_index], false};
        } else {
          return index - d_total_num_ones_[array_index];
        }
      } else {
        if constexpr (GetBit) {
          return RankResult{d_total_num_ones_[array_index], false};
        } else {
          return d_total_num_ones_[array_index];
        }
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
      utils::shareVar<size_t>(t_id == 0, result, mask);

      if constexpr (GetBit) {
        utils::shareVar<uint8_t>(has_last_word, bit_at_index, mask);
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
   * \tparam NumThreads Number of threads accessing the function. Must be less
   * than or equal to 32.
   * \param UseSamples If true, the function uses the select samples to
   * accelerate the search. Should always be true. False only used for
   * construction of the support structure.
   * \param array_index Index of the bit array to be used.
   * \param i i-th zero or one, starting from 1.
   * \param BA_offset Index in the underlying array where the bit array starts.
   * \return Position of the i-th zero/one. If i is larger than the number of
   * zeros/ones, the function returns the size of the bit array.
   */
  template <uint32_t Value, int NumThreads, bool UseSamples = true>
  __device__ [[nodiscard]] size_t select(uint32_t const array_index, size_t i,
                                         size_t const BA_offset) {
    static_assert(Value == 0 or Value == 1, "Value must be 0 or 1.");
    static_assert(NumThreads <= WS);
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

    size_t const prev_sample_index = i / RSConfig::SELECT_SAMPLE_RATE;
    size_t result =
        1;  // 1 so that "start_l1_block" is not 0 if no sample found
    size_t next_sample_pos = bit_array_.size(array_index);
    if constexpr (UseSamples) {
      if (prev_sample_index > 0) {
        if constexpr (Value == 0) {
          result =
              d_select_samples_0_[d_select_samples_0_offsets_[array_index] +
                                  prev_sample_index -
                                  1];  // -1 since pos of 0th val is not saved
          if ((prev_sample_index + 1) * RSConfig::SELECT_SAMPLE_RATE <=
              getTotalNumVals<0>(array_index)) {
            next_sample_pos =
                d_select_samples_0_[d_select_samples_0_offsets_[array_index] +
                                    prev_sample_index];
          }
        } else {
          result =
              d_select_samples_1_[d_select_samples_1_offsets_[array_index] +
                                  prev_sample_index - 1];
          if ((prev_sample_index + 1) * RSConfig::SELECT_SAMPLE_RATE <=
              getTotalNumVals<1>(array_index)) {
            next_sample_pos =
                d_select_samples_1_[d_select_samples_1_offsets_[array_index] +
                                    prev_sample_index];
          }
        }
      } else {
        if constexpr (Value == 0) {
          if (RSConfig::SELECT_SAMPLE_RATE <= getTotalNumVals<0>(array_index)) {
            next_sample_pos =
                d_select_samples_0_[d_select_samples_0_offsets_[array_index]];
          }
        } else {
          if (RSConfig::SELECT_SAMPLE_RATE <= getTotalNumVals<1>(array_index)) {
            next_sample_pos =
                d_select_samples_1_[d_select_samples_1_offsets_[array_index]];
          }
        }
      }
    }

    //  Find next L1 block where the i-th zero/one is, starting from result
    size_t const l1_offset = d_l1_offsets_[array_index];
    size_t const l2_offset = d_l2_offsets_[array_index];

    size_t const num_l1_blocks = d_num_l1_blocks_[array_index];
    size_t has_i_before = 0;

    if (num_l1_blocks > 1) {
      // Starting from 1 since 0-th entry always 0.
      size_t const start_l1_block =
          (result + RSConfig::L1_BIT_SIZE - 1) / RSConfig::L1_BIT_SIZE;
      size_t const end_l1_block =
          min(num_l1_blocks, ((next_sample_pos + RSConfig::L1_BIT_SIZE - 1) /
                              RSConfig::L1_BIT_SIZE) +
                                 1);
      if constexpr (not UseSamples) {
        assert(end_l1_block == num_l1_blocks);
        assert(start_l1_block == 1);
      }

      uint32_t const l1_blocks_per_thread =
          ((end_l1_block - start_l1_block) + NumThreads - 1) / NumThreads;

      size_t const local_start_l1_block =
          start_l1_block + local_t_id * l1_blocks_per_thread;
      size_t const local_end_l1_block =
          min(local_start_l1_block + l1_blocks_per_thread, end_l1_block);

      if (local_start_l1_block < local_end_l1_block) {
        if constexpr (Value == 0) {
          has_i_before =
              thrust::lower_bound(
                  thrust::seq, d_l1_indices_ + l1_offset + local_start_l1_block,
                  d_l1_indices_ + l1_offset + local_end_l1_block, i,
                  [&](auto const& elem, auto const& val) {
                    auto const num_bits =
                        reinterpret_cast<uintptr_t>(&elem - d_l1_indices_ -
                                                    l1_offset) *
                        RSConfig::L1_BIT_SIZE;
                    return (num_bits - elem) < val;
                  }) -
              (d_l1_indices_ + l1_offset);
        } else {
          has_i_before =
              thrust::lower_bound(
                  thrust::seq, d_l1_indices_ + l1_offset + local_start_l1_block,
                  d_l1_indices_ + l1_offset + local_end_l1_block, i) -
              (d_l1_indices_ + l1_offset);
        }
        if (has_i_before == local_end_l1_block) {
          has_i_before = 0;
        }
      }

      if constexpr (NumThreads > 1) {
        utils::shareVar<size_t>(has_i_before != 0, has_i_before, mask);
      }
    }
    if (has_i_before != 0) {
      size_t const curr_l1_block = has_i_before - 1;
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

    bool const is_last_l1_block = has_i_before == 0;
    RSConfig::L2_TYPE const l1_block_length =
        is_last_l1_block ? d_num_last_l2_blocks_[array_index]
                         : RSConfig::NUM_L2_PER_L1;
    size_t const l1_block_start =
        (result / RSConfig::L1_BIT_SIZE) * RSConfig::NUM_L2_PER_L1;
    RSConfig::L2_TYPE const l2_blocks_per_thread =
        (l1_block_length + NumThreads - 1) / NumThreads;

    size_t const start_l2_block =
        l1_block_start + local_t_id * l2_blocks_per_thread;
    size_t const end_l2_block = min(l1_block_start + l1_block_length,
                                    start_l2_block + l2_blocks_per_thread);

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
      utils::shareVar<size_t>(has_i_before != 0, has_i_before, mask);
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
        (has_i_before == 0 and is_last_l1_block)
            ? bit_array_.sizeInWords(array_index) -
                  (result / (sizeof(uint32_t) * 8))
            : RSConfig::L2_WORD_SIZE;

    size_t const word_start = result / (sizeof(uint32_t) * 8);
    result = 0;  // 0 signalizes that nothing was found
    if constexpr (NumThreads > 1) {
      for (size_t current_group_word = word_start;
           current_group_word < word_start + l2_block_length;
           current_group_word += 2 * NumThreads) {
        size_t const local_word = current_group_word + 2 * local_t_id;
        uint64_t word;
        if constexpr (Value == 0) {
          word = ~0ULL;
        } else {
          word = 0ULL;
        }
        RSConfig::L2_TYPE num_vals = 0;
        if (local_word < word_start + l2_block_length) {
          word = bit_array_.twoWords(array_index, local_word, BA_offset);
          if constexpr (Value == 0) {
            num_vals = (sizeof(uint64_t) * 8) - __popcll(word);
          } else {
            num_vals = __popcll(word);
          }
        }
        RSConfig::L2_TYPE cum_vals = 0;

        // inclusive prefix sum
        warpSum<RSConfig::L2_TYPE, NumThreads, true>(num_vals, cum_vals, mask);

        // Check if the i-th zero/one is in the current word
        RSConfig::L2_TYPE const vals_at_start = cum_vals - num_vals;
        if (cum_vals >= i and vals_at_start < i) {
          // Find the position of the i-th zero/one in the word
          // 1-indexed to distinguish from having found nothing, which is
          // 0.
          i -= vals_at_start;
          result = local_word * (sizeof(uint32_t) * 8) +
                   getNBitPos<Value, uint64_t>(i, word) + 1;
        }
        // communicate the result to all threads
        utils::shareVar<size_t>(result != 0, result, mask);
        if (result != 0) {
          break;
        }
        // if no result found, update i
        i -= cum_vals;
        utils::shareVar<size_t>(local_t_id == (NumThreads - 1), i, mask);
      }
    } else {
      RSConfig::L2_TYPE num_vals = 0;
      for (size_t j = 0; j < l2_block_length; j += 2) {
        uint64_t word;
        if constexpr (Value == 0) {
          word = ~0ULL;
        } else {
          word = 0ULL;
        }
        RSConfig::L2_TYPE current_num_vals = 0;

        word = bit_array_.twoWords(array_index, word_start + j, BA_offset);
        if constexpr (Value == 0) {
          current_num_vals = (sizeof(uint64_t) * 8) - __popcll(word);
        } else {
          current_num_vals = __popcll(word);
        }
        if ((num_vals + current_num_vals) >= i and num_vals < i) {
          i -= num_vals;
          result = (word_start + j) * (sizeof(uint32_t) * 8) +
                   getNBitPos<Value, uint64_t>(i, word) + 1;
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
  __device__ [[nodiscard]] __forceinline__ size_t
  getNumL1Blocks(uint32_t const array_index) const noexcept {
    assert(array_index < bit_array_.numArrays());
    return d_num_l1_blocks_[array_index];
  }

  /*!
   * \brief Get the number of L2 blocks for a bit array.
   * \param array_index Index of the bit array to be used.
   * \return Number of L2 blocks.
   */
  __device__ [[nodiscard]] __forceinline__ size_t
  getNumL2Blocks(uint32_t const array_index) const noexcept {
    assert(array_index < bit_array_.numArrays());
    if (array_index == bit_array_.numArrays() - 1) {
      return total_num_l2_blocks_ - d_l2_offsets_[array_index];
    }
    return d_l2_offsets_[array_index + 1] - d_l2_offsets_[array_index];
  }

  /*!
   * \brief Get the number of L2 blocks in the last L1 block for a bit array.
   * \param array_index Index of the bit array to be used.
   * \return Number of L2 blocks in the last L1 block.
   */
  __device__ [[nodiscard]] __forceinline__ size_t
  getNumLastL2Blocks(uint32_t const array_index) const noexcept {
    assert(array_index < bit_array_.numArrays());
    return d_num_last_l2_blocks_[array_index];
  }

  /*!
   * \brief Get the total number of zeros or ones in a bit array.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param array_index Index of the bit array to be used.
   * \return Total number of zeros or ones in the bit array.
   */
  template <uint32_t Value>
  __device__ [[nodiscard]] __forceinline__ size_t
  getTotalNumVals(uint32_t const array_index) const noexcept {
    static_assert(Value == 0 or Value == 1, "Value must be 0 or 1.");
    if constexpr (Value == 1) {
      return d_total_num_ones_[array_index];
    } else {
      return bit_array_.size(array_index) - d_total_num_ones_[array_index];
    }
  }

  /*!
   * \brief Get the select sample for a bit array.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the select sample to be returned.
   * \return Select sample.
   */
  template <uint32_t Value>
  __device__ [[nodiscard]] __forceinline__ size_t getSelectSample(
      uint32_t const array_index, size_t const index) const noexcept {
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
  __device__ __forceinline__ void writeL2Index(
      uint32_t const array_index, size_t const index,
      RSConfig::L2_TYPE const value) noexcept {
    assert(array_index < bit_array_.numArrays());
    assert(index < getNumL2Blocks(array_index));
    d_l2_indices_[d_l2_offsets_[array_index] + index] = value;
  }

  /*!
   * \brief Write a value to the L1 index.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L1 index to be written to.
   * \param value Value to be written.
   */
  __device__ __forceinline__ void writeL1Index(
      uint32_t const array_index, size_t const index,
      RSConfig::L1_TYPE const value) noexcept {
    assert(array_index < bit_array_.numArrays());
    assert(index < d_num_l1_blocks_[array_index]);
    d_l1_indices_[d_l1_offsets_[array_index] + index] = value;
  }

  /*!
   * \brief Write the number of ones in a bit array.
   * \param array_index Index of the bit array to be used.
   * \param value Number of ones in the bit array.
   */
  __device__ __forceinline__ void writeTotalNumOnes(
      uint32_t const array_index, size_t const value) noexcept {
    assert(array_index < bit_array_.numArrays());
    d_total_num_ones_[array_index] = value;
  }

  /*!
   * \brief Write a select sample for a bit array.
   * \tparam Value 0 for zeros, 1 for ones.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the select sample to be written to.
   * \param value Value to be written.
   */
  template <uint32_t Value>
  __device__ __forceinline__ void writeSelectSample(
      uint32_t const array_index, size_t const index,
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
  __device__ [[nodiscard]] __forceinline__ RSConfig::L1_TYPE getL1Entry(
      uint32_t const array_index, size_t const index) const noexcept {
    assert(array_index < bit_array_.numArrays());
    assert(index < d_num_l1_blocks_[array_index]);
    return d_l1_indices_[d_l1_offsets_[array_index] + index];
  }

  /*!
   * \brief Get a device pointer to an L1 entry for a bit array.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L1 entry to be returned.
   * \return Pointer to L1 entry.
   */
  __host__ [[nodiscard]] RSConfig::L1_TYPE* getL1EntryPointer(
      uint32_t const array_index, size_t const index) const noexcept;

  /*!
   * \brief Get an L2 entry for a bit array.
   * \param array_index Index of the bit array to be used.
   * \param index Local index of the L2 entry to be returned.
   * \return L2 entry.
   */
  __device__ [[nodiscard]] __forceinline__ size_t
  getL2Entry(uint32_t const array_index, size_t const index) const noexcept {
    assert(array_index < bit_array_.numArrays());
    assert(index < getNumL2Blocks(array_index));
    return d_l2_indices_[d_l2_offsets_[array_index] + index];
  }

  /*!
   * \brief Write the number of L2 blocks in the last L1 block of a bit array.
   * \param array_index Index of the bit array to be used.
   * \param value Number of L2 blocks in the last L1 block.
   */
  __device__ __forceinline__ void writeNumLastL2Blocks(
      uint32_t const array_index, uint16_t const value) noexcept {
    assert(array_index < bit_array_.numArrays());
    d_num_last_l2_blocks_[array_index] = value;
  }

  /*!
   * \brief Do a sum reduction with at most 32 threads.
   * \tparam T Type of the value to be summed.
   * \tparam NumThreads Number of threads accessing the function, must be less
   * than or equal to 32.
   * \param mask Mask representing the threads that should participate in the
   * reduction. First thread corresponds to the least significant bit.
   * \param val Value to be summed.
   * \return Sum of the values of the threads in the mask. Only valid for the
   * first thread in the mask.
   */

  template <typename T, int NumThreads>
  __device__ __forceinline__ T warpReduce(uint32_t const mask, T val) {
    static_assert(NumThreads <= WS);
#pragma unroll
    for (int offset = NumThreads / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
    }
    return val;
  }

  /*!
   * \brief Do a prefix sum with at most 32 threads.
   * \tparam T Type of the value to be summed.
   * \tparam NumThreads Number of threads accessing the function, must be less
   * than or equal to 32.
   * \tparam IsInclusive If true, the sum is an inclusive prefix sum, else it is
   * an exclusive prefix sum.
   * \param input Value to be summed.
   * \param output Sum of the values of the threads in the mask. Only valid for
   * the first thread in the mask.
   * \param mask Mask representing the threads that should participate in the
   * reduction. First thread corresponds to the least significant bit.
   */
  template <typename T, int NumThreads, bool IsInclusive>
  __device__ __forceinline__ void warpSum(T& input, T& output,
                                          uint32_t const mask) {
    static_assert(NumThreads <= WS);
    T val = input;
    uint8_t const lane = threadIdx.x % NumThreads;

#pragma unroll
    for (int offset = 1; offset < NumThreads; offset *= 2) {
      T temp = __shfl_up_sync(mask, val, offset);
      if (lane >= offset) {
        val += temp;
      }
    }

    if constexpr (IsInclusive) {
      output = val;
    } else {
      output = val - input;
    }
  }

  /*!
   * \brief Get the position of the n-th 0 or 1 bit in a word, starting from
   * the least significant bit.
   * \tparam Value 0 for zeros, 1 for ones.
   * \tparam T Type of the word, must be an integral type.
   * \param n Rank of the bit. Starts from 1.
   * \param word Word the bit is in.
   * \return Position of the n-th bit. Starts from 0.
   */
  template <uint32_t Value, typename T>
  __device__ [[nodiscard]] uint8_t getNBitPos(uint8_t n, T word) {
    static_assert(Value == 0 or Value == 1,
                  "Template parameter must be 0 or 1");
    static_assert(std::is_integral<T>::value, "T must be an integral type.");
    static_assert(sizeof(T) == 4 or sizeof(T) == 8, "T must be 4 or 8 bytes.");
    assert(n > 0);
    if constexpr (Value == 0) {
      word = ~word;
    }
    uint32_t mask = 0x0000FFFFu;
    uint8_t shift = 16u;
    uint8_t base = 0u;
    uint8_t count;
    if constexpr (sizeof(T) == 4) {
      assert(n <= 32);
    } else {
      assert(n <= 64);
      count = __popc(word & 0xFFFFFFFFu);
      if (n > count) {
        base = 32;
        n -= count;
        // move top 32 bits to bottom
        word >>= 32;
      }
    }

#pragma unroll
    for (uint8_t i = 0; i < 5; i++) {
      count = __popc(word & mask);
      if (n > count) {
        base += shift;
        shift >>= 1;
        mask |= mask << shift;
      } else {
        shift >>= 1;
        mask >>= shift;
      }
    }
    return base;
  }

  /*!
   * \brief Get an upper bound to the amount of GPU memory needed for the
   * RankSelect structure.
   * \param size Maximum size of a bit array.
   * \param num_arrays Number of bit arrays.
   * \return Amount of memory needed in bytes.
   */
  __host__ [[nodiscard]] static size_t getNeededGPUMemory(
      size_t const size, uint8_t const num_arrays) noexcept;

 private:
  // TODO: num_l1_blocks and l2 blocks not necessary
  RSConfig::L1_TYPE* d_l1_indices_ =
      nullptr; /*!< Device pointer to L1 indices for all arrays.*/
  RSConfig::L2_TYPE* d_l2_indices_ =
      nullptr; /*!< Device pointer to L2 indices for all arrays.*/

  size_t* d_l1_offsets_ = nullptr; /*!< Offsets where each L1 index for a bit
                            array starts.*/
  size_t* d_l2_offsets_ = nullptr; /*!< Offsets where each L2 index for a bit
                            array starts.*/

  uint16_t* d_num_last_l2_blocks_ = nullptr; /*!< Number of L2 blocks in the
                                   last L1 block for each bit array.*/
  size_t* d_num_l1_blocks_ =
      nullptr; /*!< Number of L1 blocks for each bit array.*/
  std::vector<size_t> num_l1_blocks_; /*!< Number of L1 blocks for all bit
                                      arrays. Not accessible from device.*/
  size_t total_num_l2_blocks_;        /*!< Total number of L2 blocks for all bit
                                         arrays.*/

  size_t* d_select_samples_0_ =
      nullptr; /*!< Sampled positions for select queries.*/
  size_t* d_select_samples_0_offsets_ = nullptr; /*!< Offsets where each array's
                                    select samples start.*/
  size_t* d_select_samples_1_ =
      nullptr; /*!< Sampled positions for select queries.*/
  size_t* d_select_samples_1_offsets_ = nullptr; /*!< Offsets where each array's
                                    select samples start.*/
  size_t* d_total_num_ones_ =
      nullptr; /*!< Total number of ones for each array.*/

  bool is_copy_; /*!< Flag to signal whether current object is a
                    copy.*/
};  // class RankSelect

namespace detail {
/*!
 * \brief Fill L2 indices and prepare L1 indices for prefix sum.
 * \param rank_select RankSelect object to fill indices for.
 * \param array_index Index of the bit array to be used.
 * \param num_last_l2_blocks Number of L2 blocks in the last L1 block.
 * \param num_l1_blocks Number of L1 blocks.
 * \param num_iters Number of iterations necessary to perform the exclusive
 * prefix sum over all L2 blocks inside an L1 block where each warp performs its
 * own sum.
 * \param bit_array_size Size of the bit array in bits.
 */
__global__ void calculateL2EntriesKernel(RankSelect rank_select,
                                         uint32_t const array_index,
                                         uint16_t const num_last_l2_blocks,
                                         size_t const num_l1_blocks,
                                         uint32_t const num_iters,
                                         size_t const bit_array_size);

/*!
 * \brief Calculate the select samples for the select support.
 * \param rank_select RankSelect object to calculate the select samples for.
 * \param array_index Index of the bit array to be used.
 * \param num_threads Total number of threads the kernel is launched with.
 * \param num_ones_samples Number of ones samples.
 * \param num_zeros_samples Number of zeros samples.
 */
__global__ void calculateSelectSamplesKernel(RankSelect rank_select,
                                             uint32_t const array_index,
                                             size_t const num_threads,
                                             size_t const num_ones_samples,
                                             size_t const num_zeros_samples);

/*!
 * \brief Helper kernel to write the total number of ones in the bit arrays
 * after the L1 indices have been calculated.
 * \param rank_select RankSelect object to calculate the total numbers of ones
 * for.
 * \param num_arrays Number of bit arrays.
 */
__global__ void addNumOnesKernel(RankSelect rank_select,
                                 uint32_t const num_arrays);
}  // namespace detail
}  // namespace ecl
