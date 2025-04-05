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

#include <omp.h>

#include <algorithm>
#include <bit_array.cuh>
#include <execution>
#include <utils.cuh>
#include <vector>

namespace ecl {

/*!
 * \brief Write words to a bit array in parallel.
 * \param bit_array The bit array to write to.
 * \param array_index The index of the bit array to write to.
 * \param words Array of words to write.
 * \param num_words The number of words to write.
 */
__global__ void writeWordsParallelKernel(BitArray bit_array, size_t array_index,
                                         uint32_t* words, size_t num_words);

/*!
 * \brief Generate random numbers in a given range, inclusive.
 * \tparam T Type of the numbers to generate.
 * \tparam UseLogDist If true, generate numbers in a logarithmic distribution.
 * \param nums_vec Vector to store the generated numbers.
 * \param min Minimum value of the range.
 * \param max Maximum value of the range.
 */
template <typename T, bool UseLogDist = false>
void generateRandomNums(std::vector<T>& nums_vec, T const min, T const max) {
  std::random_device rd;
  std::mt19937 gen(rd());  // Random number generator
  if constexpr (UseLogDist) {
    double const log_min = std::log(min);
    double const log_max = std::log(max);

    // Generate a uniform distribution between log_min and log_max
    std::uniform_real_distribution<double> log_dist(log_min, log_max);
    std::generate(nums_vec.begin(), nums_vec.end(),
                  [&]() { return static_cast<T>(std::exp(log_dist(gen))); });
  } else {
    std::uniform_int_distribution<T> dis(min, max);

    std::generate(nums_vec.begin(), nums_vec.end(), [&]() { return dis(gen); });
  }
}

/*!
 * \brief Generate a random alphabet of a given size.
 * \tparam T Type of the elements in the alphabet.
 * \param alphabet_size Size of the alphabet to generate.
 * \return Random alphabet.
 */
template <typename T>
std::vector<T> generateRandomAlphabet(size_t const alphabet_size) {
  std::vector<T> alphabet(alphabet_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
  if constexpr (sizeof(T) < 4) {
    std::vector<T> tmp(static_cast<size_t>(std::numeric_limits<T>::max()) + 1);
    std::iota(tmp.begin(), tmp.end(), 0ULL);
    std::shuffle(tmp.begin(), tmp.end(), gen);
    std::copy(tmp.begin(), tmp.begin() + alphabet_size, alphabet.begin());
  } else {
    std::unordered_set<T> unique_alphabet;
    while (unique_alphabet.size() < alphabet_size) {
      unique_alphabet.insert(dis(gen));
    }
    std::copy(unique_alphabet.begin(), unique_alphabet.end(), alphabet.begin());
  }
  std::sort(alphabet.begin(), alphabet.end());
  assert(alphabet.size() == alphabet_size);
  return alphabet;
}

/*!
 * \brief Create a random bit array with a given size and percentage of ones.
 * \param size Size of the bit array at each level.
 * \param num_levels Number of levels of the bit array.
 * \param is_adversarial If true, the distribution of ones is adversarial.
 * \param fill_rate Percentage of ones in the bit array.
 * \param one_bits_out Pointer to store the number of ones in the bit array.
 * \return Random bit array.
 */
__host__ BitArray createRandomBitArray(size_t size, uint8_t const num_levels,
                                       bool const is_adversarial = false,
                                       uint8_t const fill_rate = 50,
                                       size_t* one_bits_out = nullptr);

/*!
 * \brief Generate random alphabet and data.
 * \tparam T Type of the elements in the alphabet and data.
 * \param alphabet_size Size of the alphabet to generate.
 * \param data_size Size of the data to generate.
 * \param enforce_alphabet_size If true, enforce the alphabet size.
 * \return A pair containing the generated alphabet and data.
 */
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

/*!
 * \brief Generate random data from a given alphabet.
 * \tparam T Type of the elements in the data.
 * \param alphabet Alphabet to generate data from.
 * \param data_size Size of the data to generate.
 * \return Random data.
 */
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

/*!
 * \brief Measure the maximum memory usage of a given GPU.
 * \param stop Atomic boolean to stop the measurement.
 * \param can_start Atomic boolean to indicate when the function is ready to
 * start measuring, must be set to false.
 * \param max_memory_usage Reference to store the maximum memory usage.
 * \param GPU_index Index of the GPU to measure.
 */
void measureMemoryUsage(std::atomic_bool& stop, std::atomic_bool& can_start,
                        size_t& max_memory_usage, uint32_t const GPU_index);

/*!
 * \brief Generate random access queries.
 * \param data_size Size of the data.
 * \param num_queries Number of queries to generate.
 * \return Vector of random access queries.
 */
std::vector<size_t> generateRandomAccessQueries(size_t const data_size,
                                                size_t const num_queries);

/*!
 * \brief Generate random rank queries.
 * \tparam T Type of the elements in the queries.
 * \param data_size Size of the data.
 * \param num_queries Number of queries to generate.
 * \param alphabet Alphabet to generate queries from.
 * \return Vector of random rank queries.
 */
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

/*!
 * \brief Generate random select queries.
 * \tparam T Type of the elements in the queries.
 * \param hist Histogram of the data.
 * \param num_queries Number of queries to generate.
 * \param alphabet Alphabet to generate queries from.
 * \return Vector of random select queries.
 */
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

/*!
 * \brief Generate random data and histogram.
 * \tparam T Type of the elements in the data and histogram.
 * \param alphabet Alphabet to generate data from.
 * \param data_size Size of the data to generate.
 * \return A pair containing the generated data and histogram.
 */
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

/*!
 * \brief Generate random alphabet data and histogram.
 * \tparam T Type of the elements in the data and histogram.
 * \param alphabet_size Size of the alphabet to generate.
 * \param data_size Size of the data to generate.
 * \param hist Histogram to fill with the generated data.
 * \return A pair containing the generated alphabet and data.
 */
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

/*!
 * \brief Generate random data and rank/select queries.
 * \tparam T Type of the elements in the data and queries.
 * \param alphabet Alphabet to generate data from.
 * \param data_size Size of the data to generate.
 * \param num_queries Number of queries to generate.
 * \param rank_queries Vector to store the generated rank queries.
 * \param select_queries Vector to store the generated select queries.
 * \param rank_results Vector to store the results of the rank queries.
 * \param select_results Vector to store the results of the select queries.
 * \return Generated data.
 */
template <typename T>
std::vector<T> generateRandomDataAndRSQueries(
    std::vector<T> const& alphabet, size_t const data_size,
    size_t const num_queries, std::vector<RankSelectQuery<T>>& rank_queries,
    std::vector<RankSelectQuery<T>>& select_queries,
    std::vector<size_t>& rank_results, std::vector<size_t>& select_results) {
  assert(data_size > num_queries);
  if (alphabet.size() < kMinAlphabetSize) {
    throw std::invalid_argument("Alphabet size must be at least " +
                                std::to_string(kMinAlphabetSize));
  }

  std::unordered_map<size_t, std::unordered_map<T, size_t>> thread_hists;

  // Part 2: Generate data and histogram (parallel with thread-local histograms)
  std::vector<T> data(data_size);

  size_t const max_num_threads = omp_get_max_threads();
  std::vector<size_t> queries_per_thread(max_num_threads, 0);

#pragma omp parallel
  {
    // Thread-local random generator
    std::random_device thread_rd;
    std::mt19937 thread_gen(thread_rd() + omp_get_thread_num());
    std::uniform_int_distribution<size_t> thread_dis(0, alphabet.size() - 1);

    auto const thread_id = omp_get_thread_num();
    auto const num_threads = omp_get_num_threads();
    assert(thread_id < num_threads and thread_id >= 0);

    std::unordered_map<T, size_t> local_hist;
    size_t queries_to_generate = num_queries / num_threads;
    size_t const remaining_queries = num_queries % num_threads;
    if (static_cast<size_t>(thread_id) < remaining_queries) {
      queries_to_generate++;
    }
    queries_per_thread[thread_id] = queries_to_generate;
#pragma omp barrier
    size_t const start_query =
        std::accumulate(queries_per_thread.begin(),
                        queries_per_thread.begin() + thread_id, 0ULL);
    size_t curr_query = start_query;

    size_t const elements_to_generate =
        thread_id == num_threads - 1
            ? data_size / num_threads + data_size % num_threads
            : data_size / num_threads;

    size_t const start_element = thread_id * (data_size / num_threads);

    assert(elements_to_generate > queries_to_generate);
    size_t const iters_per_query = elements_to_generate / queries_to_generate;
    assert(start_element + elements_to_generate <= data_size);

    // Initialize histogram entries for all alphabet symbols
    for (auto const& symbol : alphabet) {
      local_hist[symbol] = 0;
    }
    for (size_t i = 0; i < elements_to_generate; i++) {
      size_t const curr_index = start_element + i;
      auto const symbol = alphabet[thread_dis(thread_gen)];

      local_hist[symbol]++;
      data[curr_index] = symbol;
      if (i % iters_per_query == 0 and
          (curr_query - start_query < queries_to_generate)) {
        rank_queries[curr_query] = RankSelectQuery<T>(curr_index, symbol);
        select_queries[curr_query] =
            RankSelectQuery<T>(local_hist[symbol], symbol);
        rank_results[curr_query] = local_hist[symbol] - 1;
        select_results[curr_query] = curr_index;
        curr_query++;
      }
    }
    assert(curr_query == start_query + queries_to_generate);
#pragma omp critical
    {
      thread_hists[thread_id] = local_hist;
    }
  }

  size_t const num_threads = thread_hists.size();
  size_t curr_thread = 0;
  size_t processed_queries = queries_per_thread[0];
  std::exclusive_scan(queries_per_thread.begin(), queries_per_thread.end(),
                      queries_per_thread.begin(), 0ULL);
  // Add hist rsults of previous threads to the queries after
  for (size_t i = processed_queries; i < num_queries; i++) {
    if (curr_thread < num_threads - 1 and
        i == queries_per_thread[curr_thread + 1]) {
      curr_thread++;
    }
    for (size_t j = 0; j < curr_thread; j++) {
      select_queries[i].index_ += thread_hists[j][select_queries[i].symbol_];
      rank_results[i] += thread_hists[j][rank_queries[i].symbol_];
    }
    processed_queries++;
  }
  assert(curr_thread == num_threads - 1);
  assert(std::all_of(
      rank_queries.data(), rank_queries.data() + num_queries,
      [&](const RankSelectQuery<T>& s) { return s.index_ < data_size; }));
  assert(std::all_of(select_queries.data(), select_queries.data() + num_queries,
                     [](const RankSelectQuery<T>& s) { return s.index_ > 0; }));
  assert(std::all_of(rank_results.data(), rank_results.data() + num_queries,
                     [&](const size_t& s) { return s < data_size; }));
  assert(std::all_of(select_results.data(), select_results.data() + num_queries,
                     [&](const size_t& s) { return s < data_size; }));

  return data;
}

/*!
 * \brief Generate random alphabet and data sizes.
 * \tparam T Type of the elements in the alphabet and data.
 * \param min_data_size Minimum size of the data.
 * \param max_data_size Maximum size of the data.
 * \param num_sizes Number of sizes to generate.
 * \return A pair containing the generated alphabet sizes and data sizes.
 */
template <typename T, bool UseLogDist = false>
std::pair<std::vector<size_t>, std::vector<size_t>>
generateRandomAlphabetAndDataSizes(size_t const min_data_size,
                                   size_t const max_data_size,
                                   size_t const num_sizes) {
  // MAx alphabet size seen on papers is ~3M
  size_t constexpr kMaxAlphabetSize = 5'000'000;
  std::vector<size_t> data_sizes(num_sizes);
  std::vector<size_t> alphabet_sizes(num_sizes);
  std::random_device rd;
  std::mt19937 gen(rd());
  if constexpr (UseLogDist) {
    std::uniform_real_distribution<double> dis_data(std::log(min_data_size),
                                                    std::log(max_data_size));
    for (size_t i = 0; i < num_sizes; i++) {
      data_sizes[i] = static_cast<size_t>(std::exp(dis_data(gen)));
      std::uniform_int_distribution<size_t> dis_alphabet(
          kMinAlphabetSize,
          std::min({data_sizes[i],
                    sizeof(T) < 8
                        ? static_cast<size_t>(std::numeric_limits<T>::max()) + 1
                        : static_cast<size_t>(std::numeric_limits<T>::max()),
                    kMaxAlphabetSize}));
      alphabet_sizes[i] = dis_alphabet(gen);
    }
  } else {
    std::uniform_int_distribution<size_t> dis_data(min_data_size,
                                                   max_data_size);
    for (size_t i = 0; i < num_sizes; i++) {
      data_sizes[i] = dis_data(gen);
      std::uniform_int_distribution<size_t> dis_alphabet(
          kMinAlphabetSize,
          std::min({data_sizes[i],
                    sizeof(T) < 8
                        ? static_cast<size_t>(std::numeric_limits<T>::max()) + 1
                        : static_cast<size_t>(std::numeric_limits<T>::max()),
                    kMaxAlphabetSize}));
      alphabet_sizes[i] = dis_alphabet(gen);
    }
  }
  return std::make_pair(alphabet_sizes, data_sizes);
}

/*!
 * \brief Read data from a file.
 * \tparam T Type of the data to read.
 * \param filename Name of the file to read from.
 * \param num_symbols Number of symbols to read.
 * \return Vector containing the read data.
 */
template <typename T>
__host__ std::vector<T> readDataFromFile(std::string const& filename,
                                         size_t const num_symbols) {
  static_assert(std::is_unsigned_v<T>, "T must be an unsigned integer type");

  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::streamsize file_size = file.tellg();
  if (file_size == -1) {
    throw std::runtime_error("Failed to get file size: " + filename);
  }
  if (num_symbols * sizeof(T) > static_cast<size_t>(file_size)) {
    throw std::runtime_error("Data size is larger than file size");
  }
  file.seekg(0, std::ios::beg);

  if (file_size % sizeof(T) != 0) {
    throw std::runtime_error("File size is not a multiple of the type size");
  }

  std::vector<T> data(num_symbols);

  if (!file.read(reinterpret_cast<char*>(data.data()),
                 num_symbols * sizeof(T))) {
    throw std::runtime_error("Failed to read file: " + filename);
  }

  return data;
}

/*!
 * \brief Convert data to a minimal alphabet.
 * \tparam T Type of the data to convert.
 * \param data Pointer to the data to convert.
 * \param data_size Size of the data.
 * \return Size of the minimal alphabet.
 */
template <typename T>
size_t convertDataToMinAlphabet(T* data, size_t const data_size) {
  std::vector<T> alphabet;
  std::unordered_set<T> alphabet_set;
#pragma omp parallel
  {
    auto const t_id = omp_get_thread_num();
    auto const num_threads = omp_get_num_threads();
    size_t const start = t_id * data_size / num_threads;
    size_t const end = t_id == num_threads - 1
                           ? data_size
                           : (t_id + 1) * data_size / num_threads;
    auto local_set = std::unordered_set<T>();
    for (size_t i = start; i < end; ++i) {
      local_set.insert(data[i]);
    }
#pragma omp critical
    {
      alphabet_set.insert(local_set.begin(), local_set.end());
    }
  }
  alphabet.assign(alphabet_set.begin(), alphabet_set.end());
  std::sort(std::execution::par, alphabet.begin(), alphabet.end());

#pragma omp parallel for
  for (size_t i = 0; i < data_size; ++i) {
    data[i] = std::distance(
        alphabet.begin(),
        std::lower_bound(alphabet.begin(), alphabet.end(), data[i]));
  }
  return alphabet.size();
}

/*!
 * \brief Convert data to a minimal alphabet.
 * \tparam T Type of the data to convert.
 * \param data Pointer to the data to convert.
 * \param data_size Size of the data.
 * \param alphabet Alphabet of the data.
 * \return Size of the minimal alphabet, same as alphabet.size().
 */
template <typename T>
size_t convertDataToMinAlphabet(T* data, size_t const data_size,
                                std::vector<T> const& alphabet) {
#pragma omp parallel for
  for (size_t i = 0; i < data_size; ++i) {
    data[i] = std::distance(
        alphabet.begin(),
        std::lower_bound(alphabet.begin(), alphabet.end(), data[i]));
  }
  return alphabet.size();
}
}  // namespace ecl