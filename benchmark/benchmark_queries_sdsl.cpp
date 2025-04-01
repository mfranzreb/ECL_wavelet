#include <omp.h>

#include <random>

#include "sdsl/wavelet_trees.hpp"

sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>>
    sdsl_byte_wt;

sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>,
            sdsl::wt_pc<sdsl::balanced_shape>::select_1_type,
            sdsl::wt_pc<sdsl::balanced_shape>::select_0_type, sdsl::int_tree<>>
    sdsl_short_wt;

size_t curr_alphabet_size = 0;
size_t curr_data_size = 0;

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

std::vector<size_t> generateRandomAccessQueries(size_t const data_size,
                                                size_t const num_queries) {
  std::vector<size_t> queries(num_queries);

#pragma omp parallel
  {
    // Thread-local random number generator
    std::random_device rd;
    // Add thread number to seed for better randomness across threads
    std::mt19937 gen(rd() + omp_get_thread_num());
    std::uniform_int_distribution<size_t> dis(0, data_size - 1);

#pragma omp for
    for (size_t i = 0; i < num_queries; i++) {
      queries[i] = dis(gen);
    }
  }

  return queries;
}

template <typename T>
struct RankSelectQuery {
  size_t index_;
  T symbol_;
};

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

template <typename T, bool DoRank>
size_t processRSQueries(std::vector<RankSelectQuery<T>> const& queries,
                        size_t const num_iters) {
  std::vector<size_t> results(queries.size());
  std::vector<size_t> times(num_iters);

  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  if constexpr (sizeof(T) == 1) {
    // warm-up
    for (size_t i = 0; i < 2; ++i) {
#pragma omp parallel for
      for (size_t j = 0; j < queries.size(); ++j) {
        if constexpr (DoRank) {
          results[j] = sdsl_byte_wt.rank(queries[j].index_, queries[j].symbol_);
        } else {
          results[j] =
              sdsl_byte_wt.select(queries[j].index_, queries[j].symbol_);
        }
      }
    }

    for (size_t i = 0; i < num_iters; ++i) {
      start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
      for (size_t j = 0; j < queries.size(); ++j) {
        if constexpr (DoRank) {
          results[j] = sdsl_byte_wt.rank(queries[j].index_, queries[j].symbol_);
        } else {
          results[j] =
              sdsl_byte_wt.select(queries[j].index_, queries[j].symbol_);
        }
      }
      end = std::chrono::high_resolution_clock::now();
      times[i] =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }
  } else {
    // warm-up
    for (size_t i = 0; i < 2; ++i) {
#pragma omp parallel for
      for (size_t j = 0; j < queries.size(); ++j) {
        if constexpr (DoRank) {
          results[j] =
              sdsl_short_wt.rank(queries[j].index_, queries[j].symbol_);
        } else {
          results[j] =
              sdsl_short_wt.select(queries[j].index_, queries[j].symbol_);
        }
      }
    }

    for (size_t i = 0; i < num_iters; ++i) {
      start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
      for (size_t j = 0; j < queries.size(); ++j) {
        if constexpr (DoRank) {
          results[j] =
              sdsl_short_wt.rank(queries[j].index_, queries[j].symbol_);
        } else {
          results[j] =
              sdsl_short_wt.select(queries[j].index_, queries[j].symbol_);
        }
      }
      end = std::chrono::high_resolution_clock::now();
      times[i] =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }
  }
  std::nth_element(times.begin(), times.begin() + times.size() / 2,
                   times.end());
  return times[times.size() / 2];
}

template <typename T>
static void BM_SDSL(size_t const data_size, size_t const alphabet_size,
                    size_t const num_queries, size_t const num_iters,
                    std::string const& output) {
  bool const rebuild_wt =
      curr_alphabet_size != alphabet_size or curr_data_size != data_size;

  static std::vector<T> alphabet;
  static std::vector<T> data;
  static std::unordered_map<T, size_t> hist;

  if (rebuild_wt) {
    alphabet = std::vector<T>(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    std::tie(data, hist) = generateRandomDataAndHist(alphabet, data_size);
    curr_alphabet_size = alphabet_size;
    curr_data_size = data_size;
  }
  auto access_queries = generateRandomAccessQueries(data_size, num_queries);
  size_t median_access_time = 0;

  std::vector<size_t> times(num_iters);
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  std::vector<T> results_access(num_queries);
  if constexpr (sizeof(T) == 1) {
    if (rebuild_wt) {
      sdsl::int_vector<8> signed_data(data.size());
#pragma omp parallel for
      for (size_t i = 0; i < data.size(); i++) {
        signed_data.set_int(8 * i, data[i], 8);
      }
      sdsl::construct_im(sdsl_byte_wt, signed_data);
      if (sdsl_byte_wt.sigma != alphabet_size) {
        std::cout << "alphabet_size does not match" << std::endl;
      }
    }

    // Warm-up
    for (size_t i = 0; i < 2; ++i) {
#pragma omp parallel for
      for (size_t j = 0; j < num_queries; ++j) {
        results_access[j] = sdsl_byte_wt[access_queries[j]];
      }
    }

    for (size_t i = 0; i < num_iters; ++i) {
      start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
      for (size_t j = 0; j < num_queries; ++j) {
        results_access[j] = sdsl_byte_wt[access_queries[j]];
      }
      end = std::chrono::high_resolution_clock::now();
      times[i] =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }
  } else {
    if (rebuild_wt) {
      sdsl::int_vector<> data64(data.size());
#pragma omp parallel for
      for (size_t i = 0; i < data.size(); i++) {
        data64.set_int(64 * i, static_cast<uint64_t>(data[i]), 64);
      }
      sdsl::construct_im(sdsl_short_wt, data64);
      if (sdsl_byte_wt.sigma != alphabet_size) {
        std::cout << "alphabet_size does not match" << std::endl;
      }
    }

    // Warm-up
    for (size_t i = 0; i < 2; ++i) {
#pragma omp parallel for
      for (size_t j = 0; j < num_queries; ++j) {
        results_access[j] = sdsl_short_wt[access_queries[j]];
      }
    }

    for (size_t i = 0; i < num_iters; ++i) {
      start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
      for (size_t j = 0; j < num_queries; ++j) {
        results_access[j] = sdsl_short_wt[access_queries[j]];
      }
      end = std::chrono::high_resolution_clock::now();
      times[i] =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }
  }
  std::nth_element(times.begin(), times.begin() + times.size() / 2,
                   times.end());
  median_access_time = times[times.size() / 2];

  auto rank_queries =
      generateRandomRankQueries(data_size, num_queries, alphabet);
  auto select_queries =
      generateRandomSelectQueries(hist, num_queries, alphabet);

  size_t const median_rank_time =
      processRSQueries<T, true>(rank_queries, num_iters);
  size_t const median_select_time =
      processRSQueries<T, false>(select_queries, num_iters);

  std::sort(rank_queries.begin(), rank_queries.end(),
            [](auto const& a, auto const& b) { return a.symbol_ < b.symbol_; });
  std::sort(select_queries.begin(), select_queries.end(),
            [](auto const& a, auto const& b) { return a.symbol_ < b.symbol_; });

  size_t const median_rank_time_sorted =
      processRSQueries<T, true>(rank_queries, num_iters);
  size_t const median_select_time_sorted =
      processRSQueries<T, false>(select_queries, num_iters);
  std::ofstream file(output, std::ios_base::app);
  file << data_size << "," << alphabet_size << "," << num_queries << ","
       << median_access_time << "," << median_rank_time << ","
       << median_select_time << "," << median_rank_time_sorted << ","
       << median_select_time_sorted << std::endl;
  file.close();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << "<num_iters> <output_dir>"
              << std::endl;
    return EXIT_FAILURE;
  }

  uint32_t const num_iters = std::stoi(argv[1]);
  std::string const output_dir = argv[2];

  size_t const data_size = 6'000'000'000;

  size_t const num_queries = 10'000'000;

  std::vector<size_t> const alphabet_sizes{
      4,    6,    8,    12,    16,    24,    32,    48,    64,   96,
      128,  192,  256,  384,   512,   768,   1024,  1536,  2048, 3072,
      4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536};

  auto out_file = output_dir + "/sdsl_queries_random_bm.csv";
  std::ofstream file(out_file);
  file << "data_size,alphabet_size,num_queries,median_access_time(mus),"
          "median_rank_time ,median_select_time,"
          "median_rank_time_sorted, median_select_time_sorted"
       << std::endl;
  file.close();

  for (auto const alphabet_size : alphabet_sizes) {
    if (alphabet_size <= 256) {
      BM_SDSL<uint8_t>(data_size, alphabet_size, num_queries, num_iters,
                       out_file);
    } else {
      BM_SDSL<uint16_t>(data_size / 2, alphabet_size, num_queries, num_iters,
                        out_file);
    }
  }
  return EXIT_SUCCESS;
}