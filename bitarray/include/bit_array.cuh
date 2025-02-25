#pragma once
#include <cstdint>
#include <vector>

#include "utils.cuh"

namespace ecl {

/*!
 * \brief Fixed size GPU bit array for usage in the wavelet tree.
 * **Important:** If you plan on accessing the data directly, note that the
 * bits are stored in reverse order in the 32-bit words. To be more precise, the
 * bits are stored as follows (for simplicity, shown for 8-bit words):
 *
 * | 7 6 5 4 3 2 1 0 | 15 14 13 12 11 10 9 8 | 23 22 21 20 19 18 17 16 | ...
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
   * \param sizes Number of bits each bit array contains. IMPORTANT: A BitArray
   * object cannot hold more than 256 bit arrays.
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

  /*!
   * \brief Move assignment operator.
   */
  __host__ BitArray& operator=(BitArray&& other) noexcept;

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

  __device__ [[nodiscard]] bool access(size_t const array_index,
                                       size_t const index,
                                       size_t const offset) const noexcept;

  /*!
   * \brief Access operator to write to a whole word of the bit array.
   * \param array_index Index of the bit array to be written to.
   * \param index Index of the word to be written to.
   * \param value Word to be written. Least significant bit corresponds to the
   * first bit of the word.
   */
  __device__ void writeWord(size_t const array_index, size_t const index,
                            uint32_t const value) noexcept;

  /*!
   * \brief Access operator to write to a whole word of the bit array.
   * \param array_index Index of the bit array to be written to.
   * \param index Index of a bit that is inside the word to be written to.
   * \param value Word to be written. Least significant bit corresponds to the
   * first bit of the word.
   */
  __device__ void writeWordAtBit(size_t const array_index, size_t const index,
                                 uint32_t const value) noexcept;

  /*!
   * \brief Access operator to write to a whole word of the bit array.
   * \param array_index Index of the bit array to be written to.
   * \param index Index of a bit that is inside the word to be written to.
   * \param value Word to be written. Least significant bit corresponds to the
   * first bit of the word.
   * \param offset Offset to the bit array to be written to.
   */
  __device__ void writeWordAtBit(size_t const array_index, size_t const index,
                                 uint32_t const value,
                                 size_t const offset) noexcept;

  __device__ size_t getOffset(size_t const array_index) const noexcept;

  /*!
   * \brief Direct access to one word of the raw data of the bit
   * array.
   * \param array_index Index of the bit array to be read from.
   * \param index Index of the word that should be returned.
   * \return index-th word of the raw bit array data. Least significant bit
   * corresponds to the first bit of the word.
   */
  __device__ [[nodiscard]] uint32_t word(size_t const array_index,
                                         size_t const index) const noexcept;

  __device__ [[nodiscard]] uint32_t word(size_t const array_index,
                                         size_t const index,
                                         size_t const offset) const noexcept;

  /*!
   * \brief Direct access to two words of the raw data of the bit
   * array.
   * \param array_index Index of the bit array to be read from.
   * \param index Index of the first word that should be returned. Must be even.
   * \return index-th and index-th + 1 words of the raw bit array data. Least
   * significant bit corresponds to the first bit of each word.
   */
  __device__ [[nodiscard]] uint64_t twoWords(size_t const array_index,
                                             size_t const index) const noexcept;

  __device__ [[nodiscard]] uint64_t twoWords(
      size_t const array_index, size_t const index,
      size_t const offset) const noexcept;

  /*!
   * \brief Direct access to one word of the raw data of the bit
   * array.
   * \param array_index Index of the bit array to be read from.
   * \param index Index of a bit that is inside the word that should be
   * returned.
   * \return index-th word of the raw bit array data. Least significant bit
   * corresponds to the first bit of the word.
   */
  __device__ [[nodiscard]] uint32_t wordAtBit(
      size_t const array_index, size_t const index) const noexcept;

  /*!
   * \brief Direct access to one word of the raw data of the bit
   * array.
   * \param word Word to be modified.
   * \param bit_index Index up to which the bits should be returned. Exclusive.
   * \return index-th word of the raw bit array data, with the bits [0,
   * bit_index) left unchanged, and all others set to 0. Least significant bit
   * corresponds to the first bit of the word.
   */
  __device__ [[nodiscard]] uint32_t partialWord(
      uint32_t word, uint8_t const bit_index) noexcept;

  __device__ [[nodiscard]] uint64_t partialTwoWords(
      uint64_t word, uint8_t const bit_index) noexcept;
  /*!
   * \brief Get the size of the bit array in
   * bits.
   * \param array_index Index of the bit array to get the size of.
   * \return Size of the bit array in bits.
   */
  __device__ [[nodiscard]] size_t size(size_t const array_index) const noexcept;

  /*! @copydoc size(size_t const) */
  __host__ [[nodiscard]] size_t sizeHost(
      size_t const array_index) const noexcept;

  /*!
   * \brief Get the size of the bit array in
   * words.
   * \param array_index Index of the bit array to get the size of.
   * \return Size of the bit array in words.
   */
  __device__ [[nodiscard]] size_t sizeInWords(
      size_t const array_index) const noexcept;

  /*!
   * \brief Get the number of bit arrays stored in the global array.
   * \return Number of bit arrays stored in the global array.
   */
  __host__ __device__ [[nodiscard]] uint8_t numArrays() const noexcept;

  __host__ [[nodiscard]] static size_t getNeededGPUMemory(
      size_t const size, uint8_t const num_arrays) noexcept;

 private:
  uint8_t num_arrays_;  /*!< Number of bit arrays stored in the global array.*/
  size_t total_size_;   /*!< Total size of the global array in words.*/
  size_t* d_bit_sizes_; /*!< Size of each array in bits.*/
  std::vector<size_t>
      bit_sizes_; /*!< Size of each array in bits. Only acessible from host.*/

  uint32_t* d_data_;  /*!< Array of 32-bit words used to store the content of
                         the bit array.*/
  size_t* d_offsets_; /*!< Array of offsets (in words) to the start of each bit
                         array.*/
  bool is_copy_;      /*!< Flag to signal whether current object is a copy.*/

};  // class BitArray

}  // namespace ecl

/******************************************************************************/