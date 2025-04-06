#include <benchmark/benchmark.h>

#include <random>

#include "ecl_wavelet/tree/wavelet_tree.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {

template <typename T>
static void BM_WaveletTreeConstruction(benchmark::State& state) {
  checkWarpSize(0);
  size_t const data_size = state.range(0);
  size_t const alphabet_size = state.range(1);

  std::vector<T> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0ULL);
  auto data = generateRandomData<T>(alphabet, data_size);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;

  // Memory usage
  {
    size_t max_memory_usage = 0;
    size_t free_mem_before_tree, total_mem;
    gpuErrchk(cudaMemGetInfo(&free_mem_before_tree, &total_mem));
    size_t tree_mem_usage = 0;
    std::atomic_bool done{false};
    std::atomic_bool can_start{false};
    std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                  std::ref(max_memory_usage), 0);
    while (not can_start) {
      std::this_thread::yield();
    }
    auto alphabet_copy = alphabet;
    auto wt = WaveletTree<T>(data.data(), data_size, std::vector<T>(), 0);
    done = true;
    t.join();
    size_t free_mem_after_tree;
    gpuErrchk(cudaMemGetInfo(&free_mem_after_tree, &total_mem));
    tree_mem_usage = free_mem_before_tree - free_mem_after_tree;
    state.counters["tree_memory_usage"] = tree_mem_usage;
    state.counters["memory_usage"] = max_memory_usage;
  }

  try {
    for (auto _ : state) {
      WaveletTree<T> wt(data.data(), data_size, std::vector<T>(), 0);
    }
  } catch (std::runtime_error& e) {
    state.SkipWithError(e.what());
  }
}

// For initializing CUDA
BENCHMARK(BM_WaveletTreeConstruction<uint8_t>)
    ->Args({100, 4})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

// First argument is the size of the data, second argument is the size of the
// alphabet, and the third argument is a boolean that indicates if the
// alphabet is minimal.
BENCHMARK(BM_WaveletTreeConstruction<uint8_t>)
    ->ArgsProduct({{8'000'000'000},
                   {4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_WaveletTreeConstruction<uint16_t>)
    ->ArgsProduct({{4'000'000'000},
                   {384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
                    12288, 16384, 24576, 32768, 49152, 65536}})
    ->Unit(benchmark::kMillisecond);

}  // namespace ecl