#include <benchmark/benchmark.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <random>

#include "test_benchmark_utils.cuh"

//? How to get max memory usage?
namespace ecl {

typedef unsigned long long cu_size_t;

template <typename T, bool isMinAlphabet, bool isPowTwo, bool UseShmem>
__global__ __launch_bounds__(
    MAX_TPB,
    MIN_BPM) void computeGlobalHistogramKernel(WaveletTree<T> tree, T* data,
                                               size_t const data_size,
                                               size_t* counts,
                                               T* const alphabet,
                                               size_t const alphabet_size,
                                               uint16_t const hists_per_block) {
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
    if constexpr (not isMinAlphabet) {
      char_data = thrust::lower_bound(thrust::seq, alphabet,
                                      alphabet + alphabet_size, char_data) -
                  alphabet;
    }
    if constexpr (UseShmem) {
      atomicAdd_block((cu_size_t*)&shared_hist[offset + char_data], size_t(1));
    } else {
      atomicAdd((cu_size_t*)&counts[char_data], size_t(1));
    }
    // TODO: could calculate code on the fly
    if constexpr (not isPowTwo) {
      if (char_data >= tree.getCodesStart()) {
        char_data = tree.encode(char_data).code_;
      }
    }
    data[i] = char_data;
  }

  if constexpr (UseShmem) {
    __syncthreads();
    // Reduce shared histograms to first one
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
__host__ void computeGlobalHistogram(bool const is_pow_two,
                                     size_t const data_size, T* d_data,
                                     T* d_alphabet, size_t* d_histogram,
                                     size_t const alphabet_size,
                                     bool const is_min_alphabet,
                                     bool const use_shmem) {
  struct cudaFuncAttributes funcAttrib;
  if (is_pow_two) {
    if (is_min_alphabet_) {
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, computeGlobalHistogramKernel<T, true, true, true>));
    } else {
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, computeGlobalHistogramKernel<T, false, true, true>));
    }
  } else {
    if (is_min_alphabet_) {
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, computeGlobalHistogramKernel<T, true, false, true>));
    } else {
      gpuErrchk(cudaFuncGetAttributes(
          &funcAttrib, computeGlobalHistogramKernel<T, false, false, true>));
    }
  }

  int const maxThreadsPerBlockHist =
      std::min(MAX_TPB, funcAttrib.maxThreadsPerBlock);

  struct cudaDeviceProp prop = getDeviceProperties();

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
  auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
  size_t const hist_size = sizeof(size_t) * alphabet_size_;

  auto const hists_per_SM = max_shmem_per_SM / hist_size;

  // auto const threads_per_hist =
  //     (max_threads_per_SM + hists_per_SM - 1) / hists_per_SM;

  // auto threads_per_warp_that_share = WS / (min_block_size /
  // threads_per_hist);

  // Find minimal block size where the same number of threads in a warp share a
  // hist
  // while (threads_per_warp_that_share ==
  //       WS / ((min_block_size / 2) / threads_per_hist)) {
  //  min_block_size /= 2;
  //}

  // Compute global_histogram and change text to min_alphabet
  size_t num_warps = (data_size + WS - 1) / WS;
  if (hists_per_SM >= MIN_BPM) {
    num_warps = std::min(
        num_warps,
        static_cast<size_t>(
            (max_threads_per_SM * prop.multiProcessorCount + WS - 1) / WS));
  }

  auto [num_blocks, threads_per_block] = getLaunchConfig(
      num_warps, std::max(WS, maxThreadsPerBlockHist), maxThreadsPerBlockHist);

  uint16_t const blocks_per_SM =
      (num_blocks + prop.multiProcessorCount - 1) / prop.multiProcessorCount;

  size_t const used_shmem =
      std::min(max_shmem_per_SM / blocks_per_SM, prop.sharedMemPerBlock);

  uint16_t const hists_per_block =
      std::min(static_cast<size_t>(threads_per_block), used_shmem / hist_size);
  // threads_per_warp_that_share = WS /
  // (threads_per_block / threads_per_hist);

  if (is_pow_two) {
    if (is_min_alphabet_) {
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, true, true, true>
            <<<num_blocks, threads_per_block, used_shmem>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, true, true, false>
            <<<num_blocks, threads_per_block>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      }
    } else {
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, false, true, true>
            <<<num_blocks, threads_per_block, used_shmem>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, false, true, false>
            <<<num_blocks, threads_per_block>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      }
    }
  } else {
    if (is_min_alphabet_) {
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, true, false, true>
            <<<num_blocks, threads_per_block, used_shmem>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, true, false, false>
            <<<num_blocks, threads_per_block>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      }
    } else {
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, false, false, true>
            <<<num_blocks, threads_per_block, used_shmem>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, false, false, false>
            <<<num_blocks, threads_per_block>>>(
                *this, d_data, data_size, d_histogram, d_alphabet,
                alphabet_size_, hists_per_block);
      }
    }
  }
  kernelCheck();
}

template <typename T>
static void BM_HistComputation(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  bool const use_shmem = state.range(2);

  auto alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto data = generateRandomData<T>(alphabet, data_size);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.use_shmem"] = use_shmem;

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

  for (auto _ : state) {
    gpuErrchk(cudaMemset(d_histogram, 0, alphabet_size * sizeof(size_t)));
    computeGlobalHistogram(isPowTwo<T>(alphabet_size), data_size, d_data,
                           d_alphabet, d_histogram, alphabet_size, true,
                           use_shmem);
  }
  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_alphabet));
  gpuErrchk(cudaFree(d_histogram));
}

template <typename T>
static void customArguments(benchmark::internal::Benchmark* b) {
  auto const min = 4;
  auto const max = std::numeric_limits<T>::max();
  T step;
  if constexpr (std::is_same_v<T, uint8_t>) {
    step = 4;
  } else {
    step = 100;
  }
  for (int64_t i = 0; i <= 1; ++i)
    for (int64_t j = min; j <= max; j += step)
      for (int64_t k = 100'000'000; k <= 4'000'000'000; k += 100'000'000)
        b->Args({k, j, i});
}

// For initializing CUDA
BENCHMARK(BM_HistComputation<uint8_t>)
    ->Args({100, 4, 1})
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

// First argument is the size of the data, second argument is the size of the
// alphabet, and the third argument is a boolean that indicates if the alphabet
// is minimal.
BENCHMARK(BM_HistComputation<uint8_t>)
    ->Apply(customArguments<uint8_t>)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_HistComputation<uint16_t>)
    ->Apply(customArguments<uint16_t>)
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl