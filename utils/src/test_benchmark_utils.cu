#include <omp.h>

#include <algorithm>
#include <bit_array.cuh>
#include <random>
#include <test_benchmark_utils.cuh>
#include <utils.cuh>
#include <vector>

namespace ecl {
__global__ void writeWordsParallelKernel(BitArray bit_array, size_t array_index,
                                         uint32_t* words, size_t num_words) {
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = thread_id; i < num_words; i += num_threads) {
    bit_array.writeWord(array_index, i, words[i]);
  }
}

__host__ BitArray createRandomBitArray(size_t size, uint8_t const num_levels) {
  BitArray ba(std::vector<size_t>(num_levels, size), false);

  auto num_words = (size + 31) / 32;
  std::vector<uint32_t> uint32_vec(num_words);
  generateRandomNums<uint32_t>(uint32_vec, 0, UINT32_MAX);

  uint32_t* d_words_arr;
  gpuErrchk(cudaMalloc(&d_words_arr, num_words * sizeof(uint32_t)));
  gpuErrchk(cudaMemcpy(d_words_arr, uint32_vec.data(),
                       num_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  auto [blocks, threads] = getLaunchConfig(
      num_words / 32 == 0 ? 1 : num_words / 32, kMinTPB, kMaxTPB);
  for (uint8_t i = 0; i < num_levels; ++i) {
    writeWordsParallelKernel<<<blocks, threads>>>(ba, i, d_words_arr,
                                                  num_words);
  }
  kernelCheck();
  gpuErrchk(cudaFree(d_words_arr));

  return ba;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> generateRandomAlphabetAndData(
    size_t alphabet_size, size_t const data_size) {
  std::vector<T> alphabet(alphabet_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
  std::generate(alphabet.begin(), alphabet.end(), [&]() { return dis(gen); });
  // Check that all elements are unique
  std::sort(alphabet.begin(), alphabet.end());
  // remove duplicates
  auto it = std::unique(alphabet.begin(), alphabet.end());
  alphabet_size = std::distance(alphabet.begin(), it);
  alphabet.resize(alphabet_size);

  std::vector<T> data(data_size);
  std::uniform_int_distribution<size_t> dis2(0, alphabet_size - 1);
  std::generate(data.begin(), data.end(),
                [&]() { return alphabet[dis2(gen)]; });

  return std::make_pair(alphabet, data);
}

void measureMemoryUsage(std::atomic_bool& stop, std::atomic_bool& can_start,
                        size_t& max_memory_usage) {
  max_memory_usage = 0;
  size_t free_bytes;
  size_t total_bytes;
  size_t start_bytes;
  gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));
  start_bytes = total_bytes - free_bytes;
  can_start = true;
  while (not stop.load()) {
    gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));
    max_memory_usage = std::max(max_memory_usage, total_bytes - free_bytes);
  }
  max_memory_usage -= start_bytes;
}

std::vector<size_t> generateRandomAccessQueries(size_t const data_size,
                                                size_t const num_queries) {
  std::vector<size_t> queries(num_queries);

#pragma omp parallel
  {
    // Thread-local random number generator
    std::random_device rd;
    // Add thread number to seed for better randomness across threads
    std::mt19937 gen(rd() + omp_get_thread_num());
    std::uniform_int_distribution<size_t> dis(0, data_size - 1);

#pragma omp for
    for (size_t i = 0; i < num_queries; i++) {
      queries[i] = dis(gen);
    }
  }

  return queries;
}

std::vector<size_t> generateRandomNums(size_t const min, size_t const max,
                                       size_t const num) {
  std::vector<size_t> nums(num);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(min, max);
  std::generate(nums.begin(), nums.end(), [&]() { return dis(gen); });
  return nums;
}
}  // namespace ecl