#include <cuda_profiler_api.h>

#include <algorithm>
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
 * \param size Size of the bit array at each level.
 * \param num_levels Number of levels of the bit array.
 * \return Random bit array.
 */
__host__ BitArray createRandomBitArray(size_t size, uint8_t const num_levels) {
  BitArray ba(std::vector<size_t>(num_levels, size), false);

  auto num_words = (size + 31) / 32;
  std::vector<uint32_t> uint32_vec(num_words);
  generateRandomNums<uint32_t>(uint32_vec, 0, UINT32_MAX);

  uint32_t* d_words_arr;
  gpuErrchk(cudaMalloc(&d_words_arr, num_words * sizeof(uint32_t)));
  gpuErrchk(cudaMemcpy(d_words_arr, uint32_vec.data(),
                       num_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  auto [blocks, threads] = getLaunchConfig(num_words / 32, 256, MAX_TPB);
  for (uint8_t i = 0; i < num_levels; ++i) {
    writeWordsParallelKernel<<<blocks, threads>>>(ba, i, d_words_arr,
                                                  num_words);
  }
  kernelCheck();
  gpuErrchk(cudaFree(d_words_arr));

  return ba;
}

}  // namespace ecl
int main(int argc, char** argv) {
  // size is first command line argument
  auto const size = std::stoul(argv[1]);
  auto const num_levels = std::stoul(argv[2]);
  auto bit_array = ecl::createRandomBitArray(size, num_levels);

  cudaProfilerStart();
  ecl::RankSelect rs(std::move(bit_array));
  cudaProfilerStop();
  return 0;
}