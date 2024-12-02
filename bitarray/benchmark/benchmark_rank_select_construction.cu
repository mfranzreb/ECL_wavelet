#include <benchmark/benchmark.h>

#include <bit_array.cuh>
#include <random>
#include <rank_select.cuh>

namespace ecl {

__global__ void writeWordsParallelKernel(BitArray bit_array, size_t array_index,
                                         uint32_t* words, size_t num_words) {
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = thread_id; i < num_words; i += num_threads) {
    bit_array.writeWord(array_index, i, words[i]);
  }
}

template <typename T>
void generateRandomNums(std::vector<T>& nums_vec, T const min, T const max) {
  std::random_device rd;
  std::mt19937 gen(rd());  // Random number generator
  std::uniform_int_distribution<T> dis(min, max);

  std::generate(nums_vec.begin(), nums_vec.end(), [&]() { return dis(gen); });
}

/*!
 * \brief Create a random bit array with a given size and percentage of ones.
 * \param size Size of the bit array.
 * \return Random bit array.
 */
__host__ BitArray createRandomBitArray(size_t size) {
  BitArray ba(std::vector<size_t>{size}, false);

  auto num_words = (size + 31) / 32;
  std::vector<uint32_t> uint32_vec(num_words);
  generateRandomNums<uint32_t>(uint32_vec, 0, UINT32_MAX);

  uint32_t* d_words_arr;
  gpuErrchk(cudaMalloc(&d_words_arr, num_words * sizeof(uint32_t)));
  gpuErrchk(cudaMemcpy(d_words_arr, uint32_vec.data(),
                       num_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  auto [blocks, threads] = getLaunchConfig(num_words / 32, 32, 1024);
  writeWordsParallelKernel<<<blocks, threads>>>(ba, 0, d_words_arr, num_words);
  kernelCheck();
  gpuErrchk(cudaFree(d_words_arr));

  return ba;
}

static void BM_RankSelectConstruction(benchmark::State& state) {
  auto const size = state.range(0);
  auto bit_array = createRandomBitArray(size);

  state.counters["param.size"] = size;

  for (auto _ : state) {
    RankSelect rs(std::move(bit_array));
  }
}

BENCHMARK(BM_RankSelectConstruction)
    ->Arg(1LL << 25)
    ->Arg(1LL << 30)
    ->Arg(1LL << 32)
    ->Arg(1LL << 33)
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl