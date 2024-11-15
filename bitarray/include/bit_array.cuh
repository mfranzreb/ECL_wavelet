#pragma once
#include <cstdint>
#include <vector>

#include "utils.cuh"

namespace ecl {

/*!
 * \brief Fixed size GPU bit array for usage in the wavelet tree.
 */
class BitArray {
 public:
  // Default empty constructor.
  BitArray() = default;

  // Deleted copy assignment.
  BitArray& operator=(BitArray const&) = delete;

  /*!
   * \brief Constructor. Creates a number of bit array that hold a specific,
   * fixed number of bits.
   * \param sizes Number of bits each bit array contains.
   */
  __host__ BitArray(std::vector<size_t> const& sizes);

  /*!
   * \brief Constructor. Creates a number of bit array that hold a specific,
   * fixed number of bits, set to the value given as parameter.
   * \param sizes Number of bits each bit array contains.
   * \param init_value Value to which the bits are set.
   */
  __host__ BitArray(std::vector<size_t> const& sizes, bool const init_value);

  /*!
   * \brief Copy constructor.
   */
  __host__ BitArray(BitArray const&);

  /*!
   * \brief Move constructor.
   */
  __host__ BitArray(BitArray&&) noexcept;
  // Destructor
  __host__ ~BitArray();

  /*!
   * \brief Access operator to read to a bit of the bit array.
   * \param array_index Index of the bit array to be read from.
   * \param index Index of the bit to be read to in the bit array.
   * \return boolean representing the bit.
   */
  __device__ [[nodiscard]] bool access(size_t const array_index,
                                       size_t const index) const noexcept;

  /*!
   * \brief Access operator to write to a whole word of the bit array.
   * \param array_index Index of the bit array to be written to.
   * \param index Index of a bit that is inside the word to be written to.
   * \param value Word to be written, where the first bit is at the left.
   */
  __device__ void write_word(size_t const array_index, size_t const index,
                             uint32_t const value) noexcept;

  /*!
   * \brief Direct access to one word of the raw data of the bit
   * vector.
   * \param array_index Index of the bit array to be read from.
   * \param index Index of a bit that is inside the word that should be
   * returned.
   * \return index-th word of the raw bit vector data.
   */
  __device__ [[nodiscard]] uint32_t word(size_t const array_index,
                                         size_t const index) const noexcept;

  /*!
   * \brief Get the size of the bit array in
   * bits.
   * \param array_index Index of the bit array to get the size of.
   * \return Size of the bit array in bits.
   */
  __host__ __device__ [[nodiscard]] size_t size(
      size_t const array_index) const noexcept;

  // TODO: add counter per copied class
  // TODO template word size
 private:
  size_t num_arrays_;   /*!< Number of bit arrays stored in the global array.*/
  size_t total_size_;   /*!< Total size of the global array in words.*/
  size_t* d_bit_sizes_; /*!< Size of each array in bits.*/
  size_t* d_sizes_;
  /*!< Size of the underlying data used to store the bits.*/  // OPT: could
                                                              // be infered
                                                              // from
                                                              // bit_sizes_
  uint32_t* d_data_;  /*!< Array of 32-bit words used to store the content of
                         the bit array.*/
  size_t* d_offsets_; /*!< Array of offsets (in words) to the start of each bit
                         array.*/
  bool is_copy_;      /*!< Flag to signal whether current object is a copy.*/

};  // class BitArray

}  // namespace ecl

/******************************************************************************/