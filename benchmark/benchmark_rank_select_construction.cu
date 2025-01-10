#include <benchmark/benchmark.h>

#include <bit_array.cuh>
#include <random>
#include <rank_select.cuh>

#include "test_benchmark_utils.cuh"

namespace ecl {

static void BM_RankSelectConstruction(benchmark::State& state) {
  checkWarpSize(0);
  auto const size = state.range(0);
  auto bit_array = createRandomBitArray(size, 1);

  state.counters["param.size"] = size;

  // Memory usage
  {
    size_t max_memory_usage = 0;
    std::atomic_bool done{false};
    std::atomic_bool can_start{false};
    std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                  std::ref(max_memory_usage));
    while (not can_start) {
      std::this_thread::yield();
    }
    auto ba = createRandomBitArray(size, 1);
    RankSelect rs(std::move(ba), 0);
    done = true;
    t.join();
    state.counters["memory_usage"] = max_memory_usage;
  }

  for (auto _ : state) {
    state.PauseTiming();
    auto ba_copy = bit_array;
    state.ResumeTiming();
    RankSelect rs(std::move(ba_copy), 0);
  }
}

BENCHMARK(BM_RankSelectConstruction)
    ->Arg(1LL << 4)
    ->Arg(1LL << 30)
    ->Arg(1LL << 32)
    ->Arg(1LL << 33)
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl