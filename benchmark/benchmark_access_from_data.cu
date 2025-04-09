#include <random>

#include "ecl_wavelet/tree/wavelet_tree.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {

template <typename T>
static void BM_Access(T const* data, size_t const data_size,
                      std::vector<size_t> const& num_queries,
                      int const GPU_index, int const num_iters,
                      std::string const& output) {
  std::vector<size_t> times(num_iters);
  for (auto const query_num : num_queries) {
    if (query_num > data_size) {
      std::cerr << "Query size is larger than the data size, skipping..."
                << std::endl;
      continue;
    }
    auto queries = utils::generateRandomAccessQueries(data_size, query_num);

    WaveletTree<T> wt(data, data_size, std::vector<T>{}, GPU_index);

    for (auto const pin_memory : {false, true}) {
      if (pin_memory) {
        gpuErrchk(cudaHostRegister(queries.data(), query_num * sizeof(size_t),
                                   cudaHostRegisterPortable));
      }
      // warmup
      for (int i = 0; i < 5; ++i) {
        auto results = wt.access(queries.data(), query_num);
      }
      for (int i = 0; i < num_iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = wt.access(queries.data(), query_num);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
      }
      // Get median and output
      std::nth_element(times.begin(), times.begin() + times.size() / 2,
                       times.end());
      std::ofstream out(output, std::ios_base::app);
      out << data_size << "," << query_num << "," << pin_memory << ","
          << times[times.size() / 2] << std::endl;
      out.close();
      if (pin_memory) {
        gpuErrchk(cudaHostUnregister(queries.data()));
      }
    }
  }
}

}  // namespace ecl

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <GPU_index> <num_iters> <input_dir> <output_dir>"
              << std::endl;
    return EXIT_FAILURE;
  }

  uint32_t const GPU_index = std::stoi(argv[1]);
  uint32_t const num_iters = std::stoi(argv[2]);
  std::string const input_dir = argv[3];
  std::string const output_dir = argv[4];

  ecl::utils::checkWarpSize(GPU_index);

  std::vector<size_t> const data_sizes = {1ULL << 28, 1ULL << 29, 1ULL << 30,
                                          1ULL << 31, 1ULL << 32, 1ULL << 33,
                                          1ULL << 34};

  std::vector<size_t> const num_queries = {100'000, 500'000, 1'000'000,
                                           5'000'000, 10'000'000};

  std::vector<std::string> const data_files = {
      input_dir + "/dna.txt", input_dir + "/prot.txt",
      input_dir + "/common_crawl.txt", input_dir + "/russian_CC.txt"};

  for (auto const& data_file : data_files) {
    std::string const output =
        output_dir + "/access_" + "GPU_" + std::to_string(GPU_index) + "_" +
        data_file.substr(data_file.find_last_of("/") + 1);
    std::ofstream out(output);
    out << "data_size,num_queries,pin_memory,time" << std::endl;
    out.close();

    for (auto const data_size : data_sizes) {
      if (data_file == input_dir + "/russian_CC.txt") {
        auto const data =
            ecl::utils::readDataFromFile<uint16_t>(data_file, data_size);

        ecl::BM_Access<uint16_t>(data.data(), data_size, num_queries, GPU_index,
                                 num_iters, output);
      } else {
        auto const data =
            ecl::utils::readDataFromFile<uint8_t>(data_file, data_size);

        ecl::BM_Access<uint8_t>(data.data(), data_size, num_queries, GPU_index,
                                num_iters, output);
      }
    }
  }
  return EXIT_SUCCESS;
}