#pragma once
#include <cstdint>

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

  // Deleted move constructor.
  BitArray(BitArray&&) = delete;

  // Deleted move assignment.
  BitArray& operator=(BitArray&&) = delete;

  /*!
   * \brief Constructor. Creates a bit array that holds a specific, fixed
   * number of bits.
   * \param size Number of bits the bit array contains.
   */
  __host__ BitArray(size_t const size);

  /*!
   * \brief Constructor. Creates a bit array that holds a specific, fixed
   * number of bits, set to the value given as parameter.
   * \param size Number of bits the bit array contains.
   * \param init_value Value to which the bits are set.
   */
  __host__ BitArray(size_t const size, bool const init_value);

  /*!
   * \brief Copy constructor.
   */
  __host__ BitArray(BitArray const&);

  // Destructor
  __host__ ~BitArray();

  /*!
   * \brief Access operator to read to a bit of the bit array.
   * \param index Index of the bit to be read to in the bit array.
   * \return boolean representing the bit.
   */
  __device__ [[nodiscard]] bool access(size_t const index) const noexcept;

  /*!
   * \brief Access operator to write to a whole word of the bit array.
   * \param index Index of a bit that is inside the word to be written to.
   * \param value Word to be written, where the first bit is at the left.
   */
  __device__ void write_word(size_t const index, uint32_t const value) noexcept;

  /*!
   * \brief Direct access to one word of the raw data of the bit
   * vector.
   * \param index Index of a bit that is inside the word that should be
   * returned.
   * \return index-th word of the raw bit vector data.
   */
  __device__ uint32_t word(size_t const index) const noexcept;

  /*!
   * \brief Get the size of the bit array in
   * bits.
   * \return Size of the bit array in bits.
   */
  __host__ __device__ [[nodiscard]] size_t size() const noexcept;

 private:
  size_t bit_size_ = 0; /*!< Size of the array in bits.*/
  size_t size_ = 0;  /*!< Size of the underlying data used to store the bits.*/
  uint32_t* d_data_; /*!< Array of 32-bit words used to store the content of the
                        bit array.*/
  bool is_copy_;     /*!< Flag to signal whether current object is a copy.*/

};  // class BitArray

}  // namespace ecl

/******************************************************************************/