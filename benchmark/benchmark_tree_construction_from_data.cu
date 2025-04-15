#include <omp.h>

#include <random>

#include "ecl_wavelet/tree/wavelet_tree.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {

template <typename T>
static void BM_Construction(T* data, size_t const data_size,
                            int const GPU_index, int const num_iters,
                            std::string const& output) {
  std::vector<size_t> times(num_iters);

  // using min alphabet as in the PWM paper
  size_t const alphabet_size = utils::convertDataToMinAlphabet(data, data_size);

  // Memory usage
  size_t max_memory_usage = 0;
  size_t free_mem_before_tree, total_mem;
  gpuErrchk(cudaMemGetInfo(&free_mem_before_tree, &total_mem));
  size_t tree_mem_usage = 0;
  {
    std::atomic_bool done{false};
    std::atomic_bool can_start{false};
    std::thread t(utils::measureMemoryUsage, std::ref(done),
                  std::ref(can_start), std::ref(max_memory_usage), GPU_index);
    while (not can_start) {
      std::this_thread::yield();
    }
    auto wt = WaveletTree<T>(data, data_size, std::vector<T>(), GPU_index);
    done = true;
    t.join();
    size_t free_mem_after_tree;
    gpuErrchk(cudaMemGetInfo(&free_mem_after_tree, &total_mem));
    tree_mem_usage = free_mem_before_tree - free_mem_after_tree;
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
  out << data_size << "," << max_memory_usage << "," << tree_mem_usage << ","
      << times[times.size() / 2] << std::endl;
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

  ecl::utils::checkWarpSize(GPU_index);

  std::vector<size_t> const data_sizes = {1ULL << 28, 1ULL << 29, 1ULL << 30,
                                          1ULL << 31, 1ULL << 32, 1ULL << 33,
                                          1ULL << 34};

  std::vector<std::string> const data_files = {input_dir + "/dna.txt",
                                               input_dir + "/common_crawl.txt",
                                               input_dir + "/prot.txt"};

  for (auto const& data_file : data_files) {
    std::string const output =
        output_dir + "/construction_" + "GPU_" + std::to_string(GPU_index) +
        "_" + data_file.substr(data_file.find_last_of("/") + 1);
    std::ofstream out(output);
    out << "data_size,memory_usage,tree_mem,time(mus)" << std::endl;
    out.close();

    for (auto const data_size : data_sizes) {
      auto data = ecl::utils::readDataFromFile<uint8_t>(data_file, data_size);

      try {
        ecl::BM_Construction<uint8_t>(data.data(), data_size, GPU_index,
                                      num_iters, output);
      } catch (std::bad_alloc const& e) {
        std::cout << "Benchmark of tree construction failed for data size "
                  << data_size << " and file " << data_file << ": " << e.what()
                  << std::endl;
        break;
      }
    }
  }
  return EXIT_SUCCESS;
}
