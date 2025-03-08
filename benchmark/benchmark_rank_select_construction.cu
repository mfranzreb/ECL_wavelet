#include <benchmark/benchmark.h>

#include <bit_array.cuh>
#include <random>
#include <rank_select.cuh>

#include "test_benchmark_utils.cuh"

namespace ecl {

static void BM_RankSelectConstruction(benchmark::State& state) {
  checkWarpSize(0);
  auto const size = state.range(0);
  auto const num_levels = state.range(1);

  state.counters["param.size"] = size;
  state.counters["param.num_levels"] = num_levels;

  // Memory usage
  {
    size_t max_memory_usage = 0;
    std::atomic_bool done{false};
    std::atomic_bool can_start{false};
    auto ba = createRandomBitArray(size, num_levels);
    std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                  std::ref(max_memory_usage), 0);
    while (not can_start) {
      std::this_thread::yield();
    }
    RankSelect rs(std::move(ba), 0);
    done = true;
    t.join();
    state.counters["memory_usage"] = max_memory_usage;
  }

  for (auto _ : state) {
    state.PauseTiming();
    auto bit_array = createRandomBitArray(size, num_levels);
    state.ResumeTiming();
    RankSelect rs(std::move(bit_array), 0);
  }
}

BENCHMARK(BM_RankSelectConstruction)
    ->Args({500'000'000, 20})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RankSelectConstruction)
    ->ArgsProduct({{10'000'000, 50'000'000, 100'000'000, 500'000'000},
                   {1, 2, 3, 4, 5, 8, 10, 15, 20}})
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl