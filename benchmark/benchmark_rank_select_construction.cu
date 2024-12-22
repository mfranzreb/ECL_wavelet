#include <benchmark/benchmark.h>

#include <bit_array.cuh>
#include <random>
#include <rank_select.cuh>

#include "test_benchmark_utils.cuh"

namespace ecl {

static void BM_RankSelectConstruction(benchmark::State& state) {
  auto const size = state.range(0);
  auto bit_array = createRandomBitArray(size, 1);

  state.counters["param.size"] = size;

  for (auto _ : state) {
    state.PauseTiming();
    auto ba_copy = bit_array;
    state.ResumeTiming();
    RankSelect rs(std::move(ba_copy), 0);
  }
}

BENCHMARK(BM_RankSelectConstruction)
    ->Arg(1LL << 25)
    ->Arg(1LL << 30)
    ->Arg(1LL << 32)
    ->Arg(1LL << 33)
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl