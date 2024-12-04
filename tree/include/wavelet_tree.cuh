#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "rank_select.cuh"

namespace ecl {

/*!
 * \brief Struct for creating a rank or select query.
 */
template <typename T>
struct RankSelectQuery {
  size_t index_;
  T symbol_;
};

/*!
 * \brief Helper class for reducing the data with thrust::remove_if.
 */
template <typename T>
class isLongEnough {
 public:
  isLongEnough(uint8_t* d_code_lens, uint32_t l)
      : d_code_lens_(d_code_lens), l_(l) {}

  __device__ bool operator()(T const code) const {
    return d_code_lens_[code] <= l_ + 1;
  }

 private:
  uint8_t* d_code_lens_;
  uint32_t l_;
};

/*!
 * \brief Wavelet tree class.
 * \tparam T Type of the text data the tree will be built upon.
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

  /*!
   * \brief Constructor. Builds the wavelet tree from the input data.
   * \param data Input data to build the wavelet tree.
   * \param data_size Number of elements in the input data.
   * \param alphabet Alphabet of the input data. Must be sorted.
   */
  __host__ WaveletTree(T* const data, size_t data_size,
                       std::vector<T>&& alphabet);

  /*! Copy constructor*/
  __host__ WaveletTree(WaveletTree const& other);

  /*! Deleted copy assignment operator*/
  WaveletTree& operator=(const WaveletTree&) = delete;

  /*! Deleted move constructor*/
  WaveletTree(WaveletTree&&) = delete;

  /*! Deleted move assignment operator*/
  WaveletTree& operator=(WaveletTree&&) = delete;

  /*! Destructor*/
  ~WaveletTree();

  /*!
   * \brief Access the symbols at the given indices in the wavelet tree.
   * \param indices Indices of the symbols to be accessed.
   * \return Vector of symbols.
   */
  __host__ std::vector<T> access(std::vector<size_t> const& indices);

  /*!
   * \brief Rank queries on the wavelet tree. Number of occurrences of a symbol
   * up to a given index (exclusive).
   * \param queries Vector of rank queries.
   * \return Vector of ranks.
   */
  __host__ std::vector<size_t> rank(std::vector<RankSelectQuery<T>>& queries);

  /*!
   * \brief Select queries on the wavelet tree. Find the index of the k-th
   * occurrence of a symbol. Starts counting from 1.
   * \param queries Vector of select queries.
   * \return Vector of selected indices.
   */
  __host__ std::vector<size_t> select(std::vector<RankSelectQuery<T>>& queries);

  /*!
   * \brief Encodes a symbol from the alphabet.
   * \param c Symbol of the minimal alphabet to be encoded.
   * \return Encoded symbol.
   */
  __device__ Code encode(T const c);

  /*!
   * \brief Creates minimal codes for the alphabet.
   * \param alphabet Alphabet to create codes for.
   * \return Vector of codes.
   */
  __host__ static std::vector<Code> createMinimalCodes(
      std::vector<T> const& alphabet);

  /*!
   * \brief Get the size of the alphabet.
   * \return Size of the alphabet.
   */
  __device__ size_t getAlphabetSize() const;

  /*!
   * \brief Get the number of levels in the wavelet tree.
   * \return Number of levels.
   */
  __device__ size_t getNumLevels() const;

  /*!
   * \brief Get the number of occurrences of all symbols that are
   * lexicographically smaller than the i-th symbol in the alphabet. Starting
   * from 0. \param i Index of the symbol in the alphabet. \return Number of
   * occurrences of all symbols that are lexicographically smaller.
   */
  __device__ size_t getCounts(size_t i) const;

 private:
  /*!
   * \brief Struct for extended version of pointerless wavelet tree.
   */
  //? Should not be necessary, since level can be inferred from code length
  struct Count {
    size_t count_; /*!< Number of symbols that are lexicographically smaller*/
    size_t level_; /*!< Level at which the symbol has it's leaf*/
  };

  T* d_min_alphabet_;          //?Necessary?         /*!< Alphabet from
                               //[0..alphabet_size_)*/
  std::vector<T> alphabet_;    /*!< Alphabet of the wavelet tree*/
  size_t alphabet_size_;       /*!< Size of the alphabet*/
  uint8_t alphabet_start_bit_; /*!< Bit where the alphabet starts, 0 is the
                                  least significant bit*/
  Code* d_codes_; /*!< Array of codes for each symbol in the alphabet*/
  uint8_t*
      d_code_lens_; /*!< Array of code lengths for each symbol in the alphabet*/
  size_t* d_counts_;  /*!< Array of counts for each symbol*/
  size_t num_levels_; /*!< Number of levels in the wavelet tree*/
  bool is_copy_;      /*!< Flag to signal whether current object is a copy*/
};

/*!
 * \brief Kernel to compute the global histogram of the input data.
 * \details Also replaces the data with the codes instead of original symbols.
 * \param tree Wavelet tree.
 * \param data Input data.
 * \param data_size Number of elements in the input data.
 * \param counts Array to store the counts.
 * \param alphabet Alphabet of the input data.
 * \param alphabet_size Size of the alphabet.
 */
template <typename T>
__global__ void computeGlobalHistogramKernel(WaveletTree<T> tree, T* data,
                                             size_t const data_size,
                                             size_t* counts, T* const alphabet,
                                             size_t const alphabet_size);

/*!
 * \brief Kernel to fill a level of the wavelet tree.
 * \param bit_array Bit array the level will be stored in.
 * \param data Data to be filled in the level.
 * \param alphabet_start_bit Bit where the alphabet starts. 0 is the LSB.
 * \param level Level to be filled.
 */
template <typename T>
__global__ void fillLevelKernel(BitArray bit_array, T* const data,
                                uint8_t const alphabet_start_bit,
                                uint32_t const level);

/*!
 * \brief Kernel for computing access queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param indices Indices of the symbols to be accessed.
 * \param num_indices Number of indices.
 * \param results Array to store the accessed symbols.
 */
template <typename T>
__global__ void accessKernel(WaveletTree<T> tree, size_t* const indices,
                             size_t const num_indices, T* results);

/*!
 * \brief Kernel for computing rank queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param queries Array of rank queries.
 * \param num_queries Number of queries.
 * \param ranks Array to store the ranks.
 */
template <typename T>
__global__ void rankKernel(WaveletTree<T> tree,
                           RankSelectQuery<T>* const queries,
                           size_t const num_queries, size_t* const ranks);

/*!
 * \brief Kernel for computing select queries on the wavelet tree.
 * \param tree Wavelet tree.
 * \param queries Array of select queries.
 * \param num_queries Number of queries.
 * \param ranks Array to store the selected indices.
 */
template <typename T>
__global__ void selectKernel(WaveletTree<T> tree,
                             RankSelectQuery<T>* const queries,
                             size_t const num_queries, size_t* const ranks);
}  // namespace ecl