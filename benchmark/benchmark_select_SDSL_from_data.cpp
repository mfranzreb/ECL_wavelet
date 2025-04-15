#include <omp.h>

#include <execution>
#include <random>
#include <sdsl/int_vector.hpp>
#include <sdsl/wavelet_trees.hpp>
#include <unordered_map>
#include <unordered_set>

template <typename T>
struct RankSelectQuery {
  size_t index_;
  T symbol_;
};

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
std::vector<T> readDataFromFile(std::string const& filename,
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

template <typename T>
auto getSDSLTree(T const* data, size_t const data_size) {
  if constexpr (sizeof(T) == 1) {
    sdsl::int_vector<8> signed_data(data_size);
#pragma omp parallel for
    for (size_t i = 0; i < data_size; i++) {
      signed_data.set_int(8 * i, data[i], 8);
    }
    sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>>
        wt;
    sdsl::construct_im(wt, signed_data);
    return wt;
  } else {
    // Change vec to uint64_t
    sdsl::int_vector<> data64(data_size);
#pragma omp parallel for
    for (size_t i = 0; i < data_size; i++) {
      data64.set_int(64 * i, static_cast<uint64_t>(data[i]), 64);
    }
    sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>,
                sdsl::wt_pc<sdsl::balanced_shape>::select_1_type,
                sdsl::wt_pc<sdsl::balanced_shape>::select_0_type,
                sdsl::int_tree<>>
        wt;
    sdsl::construct_im(wt, data64);
    return wt;
  }
}

template <typename T>
std::vector<T> getAlphabet(T const* data, size_t const data_size) {
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
    { alphabet_set.insert(local_set.begin(), local_set.end()); }
  }
  alphabet.assign(alphabet_set.begin(), alphabet_set.end());
  std::sort(std::execution::par, alphabet.begin(), alphabet.end());
  return alphabet;
}

template <typename T>
static void BM_Select(T const* data, size_t const data_size,
                      std::vector<size_t> const& num_queries,
                      int const num_iters, std::string const& output) {
  std::vector<size_t> times(num_iters);
  for (auto const query_num : num_queries) {
    if (query_num > data_size) {
      std::cerr << "Query size is larger than the data size, skipping..."
                << std::endl;
      continue;
    }
    auto const alphabet = getAlphabet(data, data_size);

    auto wt = getSDSLTree(data, data_size);

    std::unordered_map<T, size_t> hist;
    for (auto const symbol : alphabet) {
      hist[symbol] = wt.rank(data_size, symbol);
    }

    auto queries = generateRandomSelectQueries<T>(hist, query_num, alphabet);

    std::vector<size_t> results(query_num);
    for (auto const sort_queries : {false, true}) {
      if (sort_queries) {
        std::sort(
            queries.begin(), queries.end(),
            [](auto const& a, auto const& b) { return a.symbol_ < b.symbol_; });
      }
      // warmup
      for (int i = 0; i < 5; ++i) {
#pragma omp parallel for
        for (size_t j = 0; j < query_num; ++j) {
          results[j] = wt.select(queries[j].index_, queries[j].symbol_);
        }
      }
      for (int i = 0; i < num_iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (size_t j = 0; j < query_num; ++j) {
          results[j] = wt.select(queries[j].index_, queries[j].symbol_);
        }
        auto end = std::chrono::high_resolution_clock::now();
        times[i] =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
      }
      // Get median and output
      std::nth_element(times.begin(), times.begin() + times.size() / 2,
                       times.end());
      std::ofstream out(output, std::ios_base::app);
      out << data_size << "," << query_num << "," << sort_queries << ","
          << times[times.size() / 2] << std::endl;
      out.close();
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << "<num_iters> <input_dir> <output_dir>"
              << std::endl;
    return EXIT_FAILURE;
  }

  uint32_t const num_iters = std::stoi(argv[1]);
  std::string const input_dir = argv[2];
  std::string const output_dir = argv[3];

  std::vector<size_t> const data_sizes = {1ULL << 28, 1ULL << 29, 1ULL << 30,
                                          1ULL << 31, 1ULL << 32, 1ULL << 33,
                                          1ULL << 34};

  std::vector<size_t> const num_queries = {100'000, 500'000, 1'000'000,
                                           5'000'000, 10'000'000};

  std::vector<std::string> const data_files = {input_dir + "/dna.txt",
                                               input_dir + "/prot.txt",
                                               input_dir + "/common_crawl.txt"};

  for (auto const& data_file : data_files) {
    std::string const output =
        output_dir + "/select_SDSL_" +
        data_file.substr(data_file.find_last_of("/") + 1);
    std::ofstream out(output);
    out << "data_size,num_queries,sort_queries,time(mus)" << std::endl;
    out.close();

    for (auto const data_size : data_sizes) {
      auto const data = readDataFromFile<uint8_t>(data_file, data_size);
      BM_Select<uint8_t>(data.data(), data_size, num_queries, num_iters,
                         output);
    }
  }
  return EXIT_SUCCESS;
}