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
  T* symbol_;
};

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

  ~WaveletTree();

  /*!
   * \brief Access the symbols at the given indices in the wavelet tree.
   * \param indices Indices of the symbols to be accessed.
   * \return Vector of symbols.
   */
  __host__ std::vector<T> access(std::vector<size_t> const& indices);

  /*!
   * \brief Rank queries on the wavelet tree.
   * \param queries Vector of rank queries.
   * \return Vector of ranks.
   */
  __host__ std::vector<size_t> rank(
      std::vector<RankSelectQuery<T>> const& queries);

  /*!
   * \brief Select queries on the wavelet tree.
   * \param queries Vector of select queries.
   * \return Vector of selected indices.
   */
  __host__ std::vector<size_t> select(
      std::vector<RankSelectQuery<T>> const& queries);

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

  __device__ size_t getAlphabetSize() const;

  __device__ size_t getNumLevels() const;

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

template <typename T>
__host__ WaveletTree<T> createWaveletTree(T* const data, size_t const data_size,
                                          std::vector<T>&& alphabet);

template <typename T>
__global__ void computeGlobalHistogramKernel(WaveletTree<T> tree, T* data,
                                             size_t const data_size,
                                             size_t* counts, T* const alphabet,
                                             size_t const alphabet_size);

template <typename T>
__global__ void fillLevelKernel(BitArray bit_array, T* const data,
                                uint8_t const alphabet_start_bit,
                                uint32_t const level);

/*!
 * \brief Get the alphabet of the wavelet tree.
 * \return Vector of alphabet symbols.
 */
template <typename T>
__global__ void accessKernel(WaveletTree<T> tree, size_t* const indices,
                             size_t const num_indices, T* results);

template <typename T>
__global__ void rankKernel(WaveletTree<T> tree,
                           RankSelectQuery<T>* const queries,
                           size_t const num_queries, size_t* const ranks);

template <typename T>
__global__ void selectKernel(WaveletTree<T> tree,
                             RankSelectQuery<T>* const queries,
                             size_t const num_queries, size_t* const ranks);
}  // namespace ecl