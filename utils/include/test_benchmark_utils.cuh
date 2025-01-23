#pragma once

#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>

#include <algorithm>
#include <bit_array.cuh>
#include <vector>

namespace ecl {
__global__ void writeWordsParallelKernel(BitArray bit_array, size_t array_index,
                                         uint32_t* words, size_t num_words);

template <typename T>
void generateRandomNums(std::vector<T>& nums_vec, T const min, T const max) {
  std::random_device rd;
  std::mt19937 gen(rd());  // Random number generator
  std::uniform_int_distribution<T> dis(min, max);

  std::generate(nums_vec.begin(), nums_vec.end(), [&]() { return dis(gen); });
}

/*!
 * \brief Create a random bit array with a given size and percentage of ones.
 * \param size Size of the bit array at each level.
 * \param num_levels Number of levels of the bit array.
 * \return Random bit array.
 */
__host__ BitArray createRandomBitArray(size_t size, uint8_t const num_levels);

template <typename T>
std::pair<std::vector<T>, std::vector<T>> generateRandomAlphabetAndData(
    size_t const alphabet_size, size_t const data_size,
    bool enforce_alphabet_size = false) {
  if (alphabet_size < 3) {
    throw std::invalid_argument("Alphabet size must be at least 3");
  }
  std::vector<T> alphabet(alphabet_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
  size_t filled = 0;
  do {
    std::generate(alphabet.begin() + filled, alphabet.end(),
                  [&]() { return dis(gen); });
    // Check that all elements are unique
    std::sort(alphabet.begin(), alphabet.end());
    // remove duplicates
    auto it = std::unique(alphabet.begin(), alphabet.end());
    filled = std::distance(alphabet.begin(), it);
    if (not enforce_alphabet_size and filled > 2) {
      if (filled < alphabet_size) {
        alphabet.resize(filled);
      }
      break;
    }
  } while (filled != alphabet_size);

  std::vector<T> data(data_size);
  std::uniform_int_distribution<size_t> dis2(0, filled - 1);
  std::generate(data.begin(), data.end(),
                [&]() { return alphabet[dis2(gen)]; });

  return std::make_pair(alphabet, data);
}

template <typename T>
std::vector<T> generateRandomData(std::vector<T> const& alphabet,
                                  size_t const data_size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, alphabet.size() - 1);

  std::vector<T> data(data_size);
  std::generate(data.begin(), data.end(), [&]() { return alphabet[dis(gen)]; });

  return data;
}

void measureMemoryUsage(std::atomic_bool& stop, std::atomic_bool& can_start,
                        size_t& max_memory_usage);

template <typename T>
using PinnedVector = thrust::host_vector<
    T, thrust::mr::stateless_resource_allocator<
           T, thrust::system::cuda::universal_host_pinned_memory_resource>>;

PinnedVector<size_t> generateRandomQueries(size_t const data_size,
                                           size_t const num_queries);

}  // namespace ecl