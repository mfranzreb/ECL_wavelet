#include <benchmark/benchmark.h>

#include <cub/device/device_histogram.cuh>
#include <random>

#include "test_benchmark_utils.cuh"

//? How to get max memory usage?
namespace ecl {

typedef unsigned long long cu_size_t;

template <typename T, bool UseShmem>
__global__ LB(MAX_TPB, MIN_BPM) void computeGlobalHistogramKernel(
    T* data, size_t const data_size, size_t* counts, T* const alphabet,
    size_t const alphabet_size, uint16_t const hists_per_block) {
  assert(blockDim.x % WS == 0);
  extern __shared__ size_t shared_hist[];
  size_t offset;
  if constexpr (UseShmem) {
    offset = (threadIdx.x % hists_per_block) * alphabet_size;
    for (size_t i = threadIdx.x; i < alphabet_size * hists_per_block;
         i += blockDim.x) {
      shared_hist[i] = 0;
    }
    __syncthreads();
  }

  size_t const total_threads = blockDim.x * gridDim.x;
  size_t const global_t_id = blockIdx.x * blockDim.x + threadIdx.x;
  T char_data;
  for (size_t i = global_t_id; i < data_size; i += total_threads) {
    char_data = data[i];
    if constexpr (UseShmem) {
      atomicAdd_block((cu_size_t*)&shared_hist[offset + char_data], size_t(1));
    } else {
      atomicAdd((cu_size_t*)&counts[char_data], size_t(1));
    }
  }

  if constexpr (UseShmem) {
    __syncthreads();
    // Reduce shared histograms to first one
    // TODO: Could maybe be improved
    for (size_t i = threadIdx.x; i < alphabet_size; i += blockDim.x) {
      size_t sum = shared_hist[i];
      for (size_t j = 1; j < hists_per_block; ++j) {
        sum += shared_hist[j * alphabet_size + i];
      }
      atomicAdd((cu_size_t*)&counts[i], sum);
    }
  }
}

template <typename T>
__host__ bool computeGlobalHistogram(size_t const data_size, T* d_data,
                                     T* d_alphabet, size_t* d_histogram,
                                     size_t const alphabet_size,
                                     bool const use_shmem,
                                     bool const use_small_grid) {
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                  computeGlobalHistogramKernel<T, true>));

  int const maxThreadsPerBlockHist =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  struct cudaDeviceProp prop = getDeviceProperties();

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
  auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
  size_t const hist_size = sizeof(size_t) * alphabet_size;

  auto const hists_per_SM = max_shmem_per_SM / hist_size;

  auto min_block_size =
      hists_per_SM < kMinBPM
          ? kMinTPB
          : std::max(kMinTPB,
                     static_cast<uint32_t>(max_threads_per_SM / hists_per_SM));

  // Make the minimum block size a multiple of WS
  min_block_size = ((min_block_size + WS - 1) / WS) * WS;
  // Compute global_histogram and change text to min_alphabet
  size_t num_warps = (data_size + WS - 1) / WS;
  if (use_small_grid) {
    num_warps = std::min(
        num_warps,
        static_cast<size_t>(
            (max_threads_per_SM * prop.multiProcessorCount + WS - 1) / WS));
  }

  auto [num_blocks, threads_per_block] =
      getLaunchConfig(num_warps, min_block_size, maxThreadsPerBlockHist);

  uint16_t const blocks_per_SM = max_threads_per_SM / threads_per_block;

  size_t const used_shmem =
      std::min(max_shmem_per_SM / blocks_per_SM, prop.sharedMemPerBlock);

  uint16_t const hists_per_block =
      std::min(static_cast<size_t>(threads_per_block), used_shmem / hist_size);
  if (hists_per_block > 0 and use_shmem) {
    computeGlobalHistogramKernel<T, true>
        <<<num_blocks, threads_per_block, used_shmem>>>(
            d_data, data_size, d_histogram, d_alphabet, alphabet_size,
            hists_per_block);
  } else {
    computeGlobalHistogramKernel<T, false><<<num_blocks, threads_per_block>>>(
        d_data, data_size, d_histogram, d_alphabet, alphabet_size,
        hists_per_block);
  }
  kernelCheck();
  return hists_per_block > 0;
}

template <typename T>
static void BM_HistComputation(benchmark::State& state) {
  checkWarpSize(0);

  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  bool const use_shmem = state.range(2);
  bool const use_small_grid = state.range(3);

  auto alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto data = generateRandomData<T>(alphabet, data_size);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.use_shmem"] = use_shmem;
  state.counters["param.use_small_grid"] = use_small_grid;

  T* d_data;
  T* d_alphabet;
  size_t* d_histogram;
  gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_data, data.data(), data_size * sizeof(T),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&d_alphabet, alphabet_size * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_alphabet, alphabet.data(), alphabet_size * sizeof(T),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&d_histogram, alphabet_size * sizeof(size_t)));
  bool could_use_shmem;
  for (auto _ : state) {
    gpuErrchk(cudaMemset(d_histogram, 0, alphabet_size * sizeof(size_t)));
    could_use_shmem =
        computeGlobalHistogram(data_size, d_data, d_alphabet, d_histogram,
                               alphabet_size, use_shmem, use_small_grid);
  }
  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_alphabet));
  gpuErrchk(cudaFree(d_histogram));

  state.counters["could_use_shmem"] = could_use_shmem;
}

template <typename T>
static void BM_HistComputationCUB(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);

  auto alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto data = generateRandomData<T>(alphabet, data_size);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;

  T* d_data;
  T* d_alphabet;
  size_t* d_histogram;
  gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_data, data.data(), data_size * sizeof(T),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&d_alphabet, alphabet_size * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_alphabet, alphabet.data(), alphabet_size * sizeof(T),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&d_histogram, alphabet_size * sizeof(size_t)));

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(
      d_temp_storage, temp_storage_bytes, d_data, (cu_size_t*)d_histogram,
      alphabet_size + 1, T(0), T(alphabet_size), data_size);
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  for (auto _ : state) {
    gpuErrchk(cudaMemset(d_histogram, 0, alphabet_size * sizeof(size_t)));
    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes, d_data, (cu_size_t*)d_histogram,
        alphabet_size + 1, T(0), T(alphabet_size), data_size);
  }
  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_alphabet));
  gpuErrchk(cudaFree(d_histogram));
  gpuErrchk(cudaFree(d_temp_storage));
}

template <typename T, bool IsCUB>
static void customArguments(benchmark::internal::Benchmark* b) {
  auto const min = 4;
  auto const max = std::numeric_limits<T>::max();
  T step;
  if constexpr (std::is_same_v<T, uint8_t>) {
    step = 6;
  } else {
    step = 400;
  }
  if constexpr (IsCUB) {
    for (int64_t i = 100'000'000; i <= 4'100'000'000; i += 500'000'000)
      for (int64_t j = min; j <= max; j += step) b->Args({i, j});
  } else {
    for (int64_t i = 0; i <= 1; ++i)
      for (int64_t j = min; j <= max; j += step)
        for (int64_t k = 100'000'000; k <= 4'100'000'000; k += 500'000'000)
          for (int64_t l = 0; l <= 1; ++l) b->Args({k, j, i, l});
  }
}

// For initializing CUDA
BENCHMARK(BM_HistComputation<uint16_t>)
    ->Name("Init")
    ->Args({4'100'000'000, std::numeric_limits<uint16_t>::max(), 1, 0})
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_HistComputationCUB<uint16_t>)
    ->Name("InitCUB")
    ->Args({4'100'000'000, std::numeric_limits<uint16_t>::max()})
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

// First argument is the size of the data, second argument is the size of the
// alphabet, and the third argument is a boolean that indicates if the alphabet
// is minimal.
BENCHMARK(BM_HistComputation<uint8_t>)
    ->Apply(customArguments<uint8_t, false>)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_HistComputation<uint16_t>)
    ->Apply(customArguments<uint16_t, false>)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_HistComputationCUB<uint8_t>)
    ->Apply(customArguments<uint8_t, true>)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_HistComputationCUB<uint16_t>)
    ->Apply(customArguments<uint16_t, true>)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl