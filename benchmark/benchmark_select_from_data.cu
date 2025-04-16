#include <random>

#include "ecl_wavelet/tree/wavelet_tree.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {

size_t constexpr kBenchmarkTime = 1'000'000;  // 1 second

template <typename T>
static void BM_Select(T const* data, size_t const data_size,
                      std::vector<size_t> const& num_queries,
                      int const GPU_index, std::string const& output) {
  std::vector<size_t> times;
  for (auto const query_num : num_queries) {
    if (query_num > data_size) {
      std::cerr << "Query size is larger than the data size, skipping..."
                << std::endl;
      continue;
    }

    WaveletTree<T> wt(data, data_size, std::vector<T>{}, GPU_index);

    if (not wt.isMinAlphabet()) {
      std::cerr << "Data is not in min alphabet, skipping..." << std::endl;
      continue;
    }

    auto alphabet = wt.getAlphabet();

    std::unordered_map<T, size_t> hist(alphabet.size());
    for (auto const symbol : alphabet) {
      hist[symbol] = wt.getTotalAppearances(symbol);
    }

    auto queries =
        utils::generateRandomSelectQueries(hist, query_num, alphabet);

    for (auto const sort_queries : {false, true}) {
      if (sort_queries) {
        std::sort(
            queries.begin(), queries.end(),
            [](auto const& a, auto const& b) { return a.symbol_ < b.symbol_; });
      }

      for (auto const pin_memory : {false, true}) {
        if (pin_memory) {
          gpuErrchk(cudaHostRegister(queries.data(),
                                     query_num * sizeof(RankSelectQuery<T>),
                                     cudaHostRegisterPortable));
        }
        // warmup
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 5; ++i) {
          auto results = wt.select(queries.data(), query_num);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto const warmup_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        size_t const num_iters =
            std::max<size_t>(1, kBenchmarkTime / (warmup_time / 5));
        times.resize(num_iters);
        for (size_t i = 0; i < num_iters; ++i) {
          start = std::chrono::high_resolution_clock::now();
          auto results = wt.select(queries.data(), query_num);
          end = std::chrono::high_resolution_clock::now();
          times[i] =
              std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
        }
        // Get median and output
        std::nth_element(times.begin(), times.begin() + times.size() / 2,
                         times.end());
        std::ofstream out(output, std::ios_base::app);
        out << data_size << "," << query_num << "," << pin_memory << ","
            << sort_queries << "," << times[times.size() / 2] << std::endl;
        out.close();
        if (pin_memory) {
          gpuErrchk(cudaHostUnregister(queries.data()));
        }
      }
    }
  }
}

}  // namespace ecl

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <GPU_index> <input_dir> <output_dir>"
              << std::endl;
    return EXIT_FAILURE;
  }

  uint32_t const GPU_index = std::stoi(argv[1]);
  std::string const input_dir = argv[2];
  std::string const output_dir = argv[3];

  ecl::utils::checkWarpSize(GPU_index);

  std::vector<size_t> const data_sizes = {1ULL << 28, 1ULL << 29, 1ULL << 30,
                                          1ULL << 31, 1ULL << 32, 1ULL << 33,
                                          1ULL << 34};

  std::vector<size_t> const num_queries = {100'000,   500'000,    1'000'000,
                                           5'000'000, 10'000'000, 100'000'000};

  std::vector<std::string> const data_files = {input_dir + "/dna.txt",
                                               input_dir + "/prot.txt",
                                               input_dir + "/common_crawl.txt"};

  for (auto const& data_file : data_files) {
    std::string const output =
        output_dir + "/select_" + "GPU_" + std::to_string(GPU_index) + "_" +
        data_file.substr(data_file.find_last_of("/") + 1);
    std::ofstream out(output);
    out << "data_size,num_queries,pin_memory,sort_queries,time(mus)"
        << std::endl;
    out.close();

    for (auto const data_size : data_sizes) {
      auto data = ecl::utils::readDataFromFile<uint8_t>(data_file, data_size);
      ecl::utils::convertDataToMinAlphabet(data.data(), data_size);

      try {
        ecl::BM_Select<uint8_t>(data.data(), data_size, num_queries, GPU_index,
                                output);
      } catch (std::exception const& e) {
        std::cout << "Benchmark of select failed for data size " << data_size
                  << " and file " << data_file << ": " << e.what() << std::endl;
        break;
      }
    }
  }
  return EXIT_SUCCESS;
}