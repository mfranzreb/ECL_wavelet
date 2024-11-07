#pragma once
#include <cstdint>

#include "utils.cuh"

namespace ecl {

/*!\brief Fixed size GPU bit array for usage in the wavelet tree.
 *
 *
 * **Important:** If you plan on accessing the data directly, note that the
 * bits are stored in reverse order in the 32-bit words. This saves a
 * subtraction, when shifting the bits for access. To be more precise, the
 * bits are stored as follows (for simplicity, shown for 8-bit words):
 *
 * | 7 6 5 4 3 2 1 0 | 15 14 13 12 11 10 9 8 | 23 22 21 20 19 18 17 16 | ...
 */
class BitArray {
 private:
  // Size of the bit array in bits.
  size_t bit_size_ = 0;
  // Size of the underlying data used to store the bits.
  size_t size_ = 0;
  // Array of 32-bit words used to store the content of the bit array.
  uint32_t* data_;

 public:
  // Default empty constructor.
  BitArray() = default;

  // Destructor
  ~BitArray() { gpuErrchk(cudaFree(data_)); }

  // Deleted copy constructor.
  BitArray(BitArray const&) = delete;

  // Deleted copy assignment.
  BitArray& operator=(BitArray const&) = delete;

  /*!
   * \brief Constructor. Creates a bit array that holds a specific, fixed
   * number of bits.
   * \param size Number of bits the bit array contains.
   */
  __device__ BitArray(size_t const size) noexcept
      : bit_size_(size), size_((bit_size_ >> 5) + 1) {
    gpuErrchk(cudaMalloc(&data_, size_ * sizeof(uint32_t)));
  }

  /*!
   * \brief Constructor. Creates a bit array that holds a specific, fixed
   * number of bits, set to the value given as parameter.
   * \param size Number of bits the bit array contains.
   * \param init_value Value to which the bits are set.
   */
  __device__ BitArray(size_t const size, bool const init_value) noexcept
      : bit_size_(size), size_((bit_size_ >> 5) + 1) {
    gpuErrchk(cudaMalloc(&data_, size_ * sizeof(uint32_t)));
    memset(data_, init_value ? ~(0UL) : 0UL, size_ * sizeof(uint32_t));
  }

  /*!
   * \brief Access operator to read to a bit of the bit array.
   * \param index Index of the bit to be read to in the bit array.
   * \return boolean representing the bit.
   */
  __device__ [[nodiscard]] bool operator[](size_t const index) const noexcept {
    // Get position in 32-bit word
    uint8_t const offset = index & uint32_t(0b11111);
    // Get relevant word, shift and return bit
    return (data_[index >> 5] >> offset) & 1UL;
  }

  /*!
   * \brief Direct access to one word of the raw data of the bit
   * vector.
   *
   * Note that the raw data does not contain the bits from left to right. A
   * detailed description can be found at the top of this file.
   * \param index Index of the word word that should be returned.
   * \return index-th word of the raw bit vector data.
   */
  __device__ uint32_t word(size_t const index) const noexcept {
    return data_[index];
  }

  /*!
   * \brief Get the size of the bit array in
   * bits.
   * \return Size of the bit array in bits.
   */
  __device__ [[nodiscard]] size_t size() const noexcept { return bit_size_; }

};  // class BitArray

// \}

}  // namespace ecl

/******************************************************************************/