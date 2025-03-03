#pragma once

#include <omp.h>

#include <algorithm>
#include <bit_array.cuh>
#include <utils.cuh>
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
  if (alphabet_size < kMinAlphabetSize) {
    throw std::invalid_argument("Alphabet size must be at least " +
                                std::to_string(kMinAlphabetSize));
  }

  // Part 1: Generate alphabet - this part is hard to parallelize due to its
  // sequential nature
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

  // Part 2: Generate data - this part can be parallelized
  std::vector<T> data(data_size);

#pragma omp parallel
  {
    // Create thread-local random generator
    std::random_device thread_rd;
    std::mt19937 thread_gen(thread_rd() + omp_get_thread_num());
    std::uniform_int_distribution<size_t> thread_dis(0, filled - 1);

#pragma omp for
    for (size_t i = 0; i < data_size; i++) {
      data[i] = alphabet[thread_dis(thread_gen)];
    }
  }

  return std::make_pair(alphabet, data);
}

template <typename T>
std::vector<T> generateRandomData(std::vector<T> const& alphabet,
                                  size_t const data_size) {
  std::vector<T> data(data_size);
#pragma omp parallel
  {
    // Create a thread-local random number generator
    std::random_device rd;
    std::mt19937 gen(rd() + omp_get_thread_num());  // Add thread number to seed
                                                    // for better randomness
    std::uniform_int_distribution<size_t> dis(0, alphabet.size() - 1);

#pragma omp for
    for (size_t i = 0; i < data_size; i++) {
      data[i] = alphabet[dis(gen)];
    }
  }

  return data;
}

void measureMemoryUsage(std::atomic_bool& stop, std::atomic_bool& can_start,
                        size_t& max_memory_usage);

std::vector<size_t> generateRandomAccessQueries(size_t const data_size,
                                                size_t const num_queries);

template <typename T>
std::vector<RankSelectQuery<T>> generateRandomRankQueries(
    size_t const data_size, size_t const num_queries,
    std::vector<T> const& alphabet) {
  std::vector<RankSelectQuery<T>> queries(num_queries);

#pragma omp parallel
  {
    // Thread-local random number generators
    std::random_device rd;
    std::mt19937 gen(rd() + omp_get_thread_num());  // Add thread ID to seed
    std::uniform_int_distribution<size_t> dis(0, data_size - 1);
    std::uniform_int_distribution<size_t> dis2(0, alphabet.size() - 1);

#pragma omp for
    for (size_t i = 0; i < num_queries; i++) {
      queries[i] = RankSelectQuery<T>(dis(gen), alphabet[dis2(gen)]);
    }
  }

  return queries;
}

template <typename T>
std::vector<RankSelectQuery<T>> generateRandomSelectQueries(
    std::unordered_map<T, size_t> const& hist, size_t const num_queries,
    std::vector<T> const& alphabet) {
  std::vector<RankSelectQuery<T>> queries(num_queries);
#pragma omp parallel
  {
    // Thread-local random number generator
    std::random_device rd;
    std::mt19937 gen(rd() + omp_get_thread_num());
    std::uniform_int_distribution<T> dis_alphabet(0, alphabet.size() - 1);
#pragma omp for
    for (size_t i = 0; i < num_queries; i++) {
      T symbol;
      size_t count;
      do {
        symbol = alphabet[dis_alphabet(gen)];
        count = hist.at(symbol);
      } while (count == 0);
      std::uniform_int_distribution<size_t> dis_index(1, count);
      auto index = dis_index(gen);

      queries[i] = RankSelectQuery<T>(index, symbol);
    }
  }
  return queries;
}

template <typename T>
std::pair<std::vector<T>, std::unordered_map<T, size_t>>
generateRandomDataAndHist(std::vector<T> const& alphabet,
                          size_t const data_size) {
  std::vector<T> data(data_size);
  std::unordered_map<T, size_t> hist;
  std::uniform_int_distribution<size_t> dis(0, alphabet.size() - 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::generate(data.begin(), data.end(), [&]() {
    auto const symbol = alphabet[dis(gen)];
    hist[symbol]++;
    return symbol;
  });

  // Make sure that all symbols are in the hist
  for (auto const& symbol : alphabet) {
    if (hist.find(symbol) == hist.end()) {
      hist[symbol] = 0;
    }
  }

  return std::make_pair(data, hist);
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> generateRandomAlphabetDataAndHist(
    size_t const alphabet_size, size_t const data_size,
    std::unordered_map<T, size_t>& hist) {
  if (alphabet_size < kMinAlphabetSize) {
    throw std::invalid_argument("Alphabet size must be at least " +
                                std::to_string(kMinAlphabetSize));
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
    if (filled > 2) {
      if (filled < alphabet_size) {
        alphabet.resize(filled);
      }
      break;
    }
  } while (filled != alphabet_size);

  // Initialize histogram entries for all alphabet symbols
  for (auto const& symbol : alphabet) {
    hist[symbol] = 0;
  }

  // Part 2: Generate data and histogram (parallel with thread-local histograms)
  std::vector<T> data(data_size);

#pragma omp parallel
  {
    // Thread-local random generator
    std::random_device thread_rd;
    std::mt19937 thread_gen(thread_rd() + omp_get_thread_num());
    std::uniform_int_distribution<size_t> thread_dis(0, filled - 1);

    // Thread-local histogram
    std::unordered_map<T, size_t> local_hist;

#pragma omp for
    for (size_t i = 0; i < data_size; i++) {
      auto const symbol = alphabet[thread_dis(thread_gen)];
      local_hist[symbol]++;
      data[i] = symbol;
    }

// Combine thread-local histogram with global histogram
#pragma omp critical
    {
      for (auto const& pair : local_hist) {
        hist[pair.first] += pair.second;
      }
    }
  }

  return std::make_pair(alphabet, data);
}

template <typename T>
std::pair<std::vector<size_t>, std::vector<size_t>>
generateRandomAlphabetAndDataSizes(size_t const min_data_size,
                                   size_t const max_data_size,
                                   size_t const num_sizes) {
  std::vector<size_t> data_sizes(num_sizes);
  std::vector<size_t> alphabet_sizes(num_sizes);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis_data(min_data_size, max_data_size);
  for (size_t i = 0; i < num_sizes; i++) {
    data_sizes[i] = dis_data(gen);
    std::uniform_int_distribution<size_t> dis_alphabet(
        kMinAlphabetSize,
        std::min(data_sizes[i],
                 static_cast<size_t>(std::numeric_limits<T>::max()) + 1));
    alphabet_sizes[i] = dis_alphabet(gen);
  }
  return std::make_pair(data_sizes, alphabet_sizes);
}

std::vector<size_t> generateRandomNums(size_t const min, size_t const max,
                                       size_t const num);

}  // namespace ecl