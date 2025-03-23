#include <random>
#include <utils.cuh>
#include <wavelet_tree.cuh>

#include "test_benchmark_utils.cuh"

namespace ecl {

template <typename T>
static void BM_Construction(T const* data, size_t const data_size,
                            int const GPU_index, int const num_iters,
                            std::string const& output) {
  std::vector<size_t> times(num_iters);

  // Memory usage
  size_t max_memory_usage = 0;
  {
    std::atomic_bool done{false};
    std::atomic_bool can_start{false};
    std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                  std::ref(max_memory_usage), GPU_index);
    while (not can_start) {
      std::this_thread::yield();
    }
    auto wt = WaveletTree<T>(data, data_size, std::vector<T>(), GPU_index);
    done = true;
    t.join();
  }

  // warmup
  for (int i = 0; i < 5; ++i) {
    auto wt = WaveletTree<T>(data, data_size, std::vector<T>(), GPU_index);
  }
  for (int i = 0; i < num_iters; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto wt = WaveletTree<T>(data, data_size, std::vector<T>(), GPU_index);
    auto end = std::chrono::high_resolution_clock::now();
    times[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }
  // Get median and output
  std::nth_element(times.begin(), times.begin() + times.size() / 2,
                   times.end());
  std::ofstream out(output, std::ios_base::app);
  out << data_size << "," << max_memory_usage << "," << times[times.size() / 2]
      << std::endl;
  out.close();
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

  ecl::checkWarpSize(GPU_index);

  std::vector<size_t> const data_sizes = {2ULL << 28, 2ULL << 29, 2ULL << 30,
                                          2ULL << 31, 2ULL << 32, 2ULL << 33,
                                          2ULL << 34};

  std::vector<std::string> const data_files = {
      input_dir + "/dna.txt", input_dir + "/common_crawl.txt",
      input_dir + "/prot.txt", input_dir + "/russian_CC.txt"};

  for (auto const& data_file : data_files) {
    std::string const output =
        output_dir + "/construction_" + "GPU_" + std::to_string(GPU_index) +
        "_" + data_file.substr(data_file.find_last_of("/") + 1);
    std::ofstream out(output);
    out << "data_size,memory_usage,time" << std::endl;
    out.close();

    for (auto const data_size : data_sizes) {
      if (data_file == input_dir + "/russian_CC.txt") {
        auto const data = ecl::readDataFromFile<uint16_t>(data_file, data_size);

        ecl::BM_Construction<uint16_t>(data.data(), data_size, GPU_index,
                                       num_iters, output);
      } else {
        auto const data = ecl::readDataFromFile<uint8_t>(data_file, data_size);

        ecl::BM_Construction<uint8_t>(data.data(), data_size, GPU_index,
                                      num_iters, output);
      }
    }
  }
  return EXIT_SUCCESS;
}
