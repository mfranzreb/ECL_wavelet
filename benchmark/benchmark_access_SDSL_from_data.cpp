#include <random>
#include <sdsl/int_vector.hpp>
#include <sdsl/wavelet_trees.hpp>
#include <omp.h>

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
std::vector<T> readDataFromFile(std::string const& filename) {
  static_assert(std::is_unsigned_v<T>, "T must be an unsigned integer type");

  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (file_size % sizeof(T) != 0) {
    throw std::runtime_error("File size is not a multiple of the type size");
  }

  std::vector<T> data(file_size / sizeof(T));

  if (!file.read(reinterpret_cast<char*>(data.data()), file_size)) {
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
static void BM_Access(T const* data, size_t const data_size,
                      std::vector<size_t> const& num_queries,
                      int const num_iters, std::string const& output) {
  std::vector<size_t> times(num_iters);
  for (auto const query_num : num_queries) {
    if (query_num > data_size) {
      std::cerr << "Query size is larger than the data size, skipping..."
                << std::endl;
      continue;
    }
    auto queries = generateRandomAccessQueries(data_size, query_num);

    auto wt = getSDSLTree(data, data_size);

    std::vector<T> results(query_num);
    // warmup
    for (int i = 0; i < 5; ++i) {
#pragma omp parallel for
      for (size_t j = 0; j < query_num; ++j) {
        results[j] = wt[queries[j]];
      }
    }
    for (int i = 0; i < num_iters; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
      for (size_t j = 0; j < query_num; ++j) {
        results[j] = wt[queries[j]];
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
    out << data_size << "," << query_num << "," << times[times.size() / 2]
        << std::endl;
    out.close();
  }
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << "<num_iters> <input_dir> <output_dir>"
              << std::endl;
    return EXIT_FAILURE;
  }

  uint32_t const num_iters = std::stoi(argv[1]);
  std::string const input_dir = argv[2];
  std::string const output_dir = argv[3];

  std::vector<size_t> const data_sizes = {500'000'000, 1'000'000'000,
                                          2'000'000'000};

  std::vector<size_t> const num_queries = {100'000, 500'000, 1'000'000,
                                           5'000'000, 10'000'000};

  std::vector<std::string> const data_files = {
      input_dir + "/dna.txt", input_dir + "/common_crawl.txt",
      input_dir + "/prot.txt", input_dir + "/russian_CC.txt"};

  for (auto const& data_file : data_files) {
    std::string const output =
        output_dir + "/access_" +
        data_file.substr(data_file.find_last_of("/") + 1);
    std::ofstream out(output);
    out << "data_size,num_queries,time" << std::endl;
    out.close();

    for (auto const data_size : data_sizes) {
      if (data_file == input_dir + "/russian_CC.txt") {
        auto const data = readDataFromFile<uint16_t>(data_file);
        if (data_size > data.size()) {
          std::cerr << "Data size is larger than the file size, skipping..."
                    << std::endl;
          continue;
        }
        BM_Access<uint16_t>(data.data(), data_size, num_queries, num_iters,
                            output);
      } else {
        auto const data = readDataFromFile<uint8_t>(data_file);
        if (data_size > data.size()) {
          std::cerr << "Data size is larger than the file size, skipping..."
                    << std::endl;
          continue;
        }
        BM_Access<uint8_t>(data.data(), data_size, num_queries, num_iters,
                           output);
      }
    }
  }
  return EXIT_SUCCESS;
}