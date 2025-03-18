#include <benchmark/benchmark.h>

#include <bit_array.cuh>
#include <random>
#include <rank_select.cuh>

#include "test_benchmark_utils.cuh"

namespace ecl {

template <int Value>
__global__ LB(MAX_TPB, MIN_BPM) void binaryRankKernel(
    RankSelect rs, size_t* const queries, size_t* results,
    size_t const num_queries, size_t const num_threads) {
  size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < num_queries; i += num_threads) {
    results[i] = rs.rank<1, 0, Value>(0, queries[i], 0);
  }
}

template <int Value>
__global__ LB(MAX_TPB, MIN_BPM) void binarySelectKernel(
    RankSelect rs, size_t* const ranks, size_t* results,
    size_t const num_queries, size_t const num_threads) {
  size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < num_queries; i += num_threads) {
    results[i] = rs.select<1, 0, Value>(0, ranks[i], 0);
  }
}

static void BM_RankSelectConstruction(benchmark::State& state) {
  checkWarpSize(0);
  size_t const size = state.range(0);
  bool const is_adversarial = state.range(1);
  auto const fill_rate = state.range(2);

  state.counters["param.size"] = size;
  state.counters["param.is_adversarial"] = is_adversarial;
  state.counters["param.fill_rate"] = fill_rate;

  // Memory usage
  {
    size_t max_memory_usage = 0;
    std::atomic_bool done{false};
    std::atomic_bool can_start{false};
    auto ba = createRandomBitArray(size, 1, is_adversarial, fill_rate);
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
    auto bit_array = createRandomBitArray(size, 1, is_adversarial, fill_rate);
    state.ResumeTiming();
    RankSelect rs(std::move(bit_array), 0);
  }
}

template <int Value>
static void BM_binaryRank(benchmark::State& state) {
  checkWarpSize(0);
  size_t const size = state.range(0);
  bool const is_adversarial = state.range(1);
  auto const fill_rate = state.range(2);
  size_t const num_queries = state.range(3);

  state.counters["param.size"] = size;
  state.counters["param.is_adversarial"] = is_adversarial;
  state.counters["param.fill_rate"] = fill_rate;
  state.counters["param.num_queries"] = num_queries;
  auto bit_array = createRandomBitArray(size, 1, is_adversarial, fill_rate);
  RankSelect rs(std::move(bit_array), 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dist(0, size - 1);
  std::vector<size_t> queries(num_queries);
  std::generate(queries.begin(), queries.end(), [&]() { return dist(gen); });

  size_t* d_queries;
  size_t* d_results;
  gpuErrchk(cudaMalloc(&d_queries, num_queries * sizeof(size_t)));
  gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_queries, queries.data(), num_queries * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  struct cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, binaryRankKernel<Value>);
  auto max_block_size =
      std::min(kMaxTPB, static_cast<uint32_t>(attr.maxThreadsPerBlock));

  max_block_size = findLargestDivisor(kMaxTPB, max_block_size);

  auto const& prop = getDeviceProperties();

  size_t const num_warps = std::min(
      num_queries,
      static_cast<size_t>(
          (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / WS));
  auto const [num_blocks, block_size] =
      getLaunchConfig(num_warps, kMinTPB, max_block_size);

  for (auto _ : state) {
    gpuErrchk(cudaEventRecord(start));
    binaryRankKernel<Value><<<num_blocks, block_size>>>(
        rs, d_queries, d_results, num_queries, num_blocks * block_size);
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    state.SetIterationTime(static_cast<double>(milliseconds) / 1000);
  }

  // Check that all results are valid.
  std::vector<size_t> results(num_queries);
  gpuErrchk(cudaMemcpy(results.data(), d_results, num_queries * sizeof(size_t),
                       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_queries));
  gpuErrchk(cudaFree(d_results));
  for (size_t i = 0; i < num_queries; ++i) {
    if (results[i] >= size) {
      std::cerr << "Invalid result: " << results[i] << std::endl;
      std::exit(1);
    }
  }
}

template <int Value>
static void BM_binarySelect(benchmark::State& state) {
  checkWarpSize(0);
  size_t const size = state.range(0);
  bool const is_adversarial = state.range(1);
  auto const fill_rate = state.range(2);
  size_t const num_queries = state.range(3);

  state.counters["param.size"] = size;
  state.counters["param.is_adversarial"] = is_adversarial;
  state.counters["param.fill_rate"] = fill_rate;
  state.counters["param.num_queries"] = num_queries;
  size_t one_bits = 0;
  auto bit_array =
      createRandomBitArray(size, 1, is_adversarial, fill_rate, &one_bits);
  RankSelect rs(std::move(bit_array), 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<size_t> queries(num_queries);
  if constexpr (Value == 1) {
    std::uniform_int_distribution<size_t> dist(1, one_bits);
    std::generate(queries.begin(), queries.end(), [&]() { return dist(gen); });
  } else {
    std::uniform_int_distribution<size_t> dist(1, size - one_bits);
    std::generate(queries.begin(), queries.end(), [&]() { return dist(gen); });
  }

  size_t* d_queries;
  size_t* d_results;
  gpuErrchk(cudaMalloc(&d_queries, num_queries * sizeof(size_t)));
  gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_queries, queries.data(), num_queries * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  struct cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, binaryRankKernel<Value>);
  auto max_block_size =
      std::min(kMaxTPB, static_cast<uint32_t>(attr.maxThreadsPerBlock));

  max_block_size = findLargestDivisor(kMaxTPB, max_block_size);

  auto const& prop = getDeviceProperties();

  size_t const num_warps = std::min(
      num_queries,
      static_cast<size_t>(
          (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / WS));
  auto const [num_blocks, block_size] =
      getLaunchConfig(num_warps, kMinTPB, max_block_size);

  for (auto _ : state) {
    gpuErrchk(cudaEventRecord(start));
    binaryRankKernel<Value><<<num_blocks, block_size>>>(
        rs, d_queries, d_results, num_queries, num_blocks * block_size);
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    state.SetIterationTime(static_cast<double>(milliseconds) / 1000);
  }

  // Check that all results are valid.
  std::vector<size_t> results(num_queries);
  gpuErrchk(cudaMemcpy(results.data(), d_results, num_queries * sizeof(size_t),
                       cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_queries));
  gpuErrchk(cudaFree(d_results));
  for (size_t i = 0; i < num_queries; ++i) {
    if (results[i] >= size) {
      std::cerr << "Invalid result: " << results[i] << std::endl;
      std::exit(1);
    }
  }
}

BENCHMARK(BM_RankSelectConstruction)
    ->Args({500'000'000, 0, 50})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RankSelectConstruction)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binaryRank<0>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90},
                   {100'000'000}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binaryRank<1>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90},
                   {100'000'000}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binarySelect<0>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90},
                   {100'000'000}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_binarySelect<1>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90},
                   {100'000'000}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace ecl