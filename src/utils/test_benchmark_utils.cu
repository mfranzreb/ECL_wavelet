/*
BSD 3-Clause License

Copyright (c) 2025, Marco Franzreb, Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <omp.h>

#include <algorithm>
#include <random>
#include <vector>

#include "ecl_wavelet/bitarray/bit_array.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {
__global__ void writeWordsParallelKernel(BitArray bit_array, size_t array_index,
                                         uint32_t* words, size_t num_words) {
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = thread_id; i < num_words; i += num_threads) {
    bit_array.writeWord(array_index, i, words[i]);
  }
}

__host__ BitArray createRandomBitArray(size_t size, uint8_t const num_levels,
                                       bool const is_adversarial,
                                       uint8_t const fill_rate,
                                       size_t* one_bits_out) {
  assert(fill_rate <= 100);
  BitArray ba(std::vector<size_t>(num_levels, size), false);

  auto num_words = (size + 31) / 32;
  std::vector<uint32_t> bits(num_words);

  size_t one_bits = 0;
  if (!is_adversarial) {
#pragma omp parallel reduction(+ : one_bits)
    {
      std::random_device rd;
      std::mt19937 gen(rd() ^ omp_get_thread_num());  // Thread-local generator
      std::uniform_int_distribution<size_t> bit_dist(0, 99);

#pragma omp for
      for (size_t i = 0; i < num_words; i++) {
        uint32_t word = 0;
        for (size_t j = 0; j < 32; ++j) {
          bool const flip_bit =
              (static_cast<uint32_t>(bit_dist(gen)) < fill_rate);
          one_bits += flip_bit ? 1 : 0;
          word |= static_cast<uint32_t>(flip_bit) << j;
        }
        bits[i] = word;
      }
    }
  } else {
    size_t const split_index = (num_words / 100) * (100 - fill_rate) / 32;

#pragma omp parallel reduction(+ : one_bits)
    {
      std::random_device rd;
      std::mt19937 gen(rd() ^ omp_get_thread_num());
      std::uniform_int_distribution<size_t> bit_dist(0, 99);

#pragma omp for
      for (size_t i = 0; i < split_index; ++i) {
        uint32_t word = 0;
        for (size_t j = 0; j < 32; ++j) {
          bool const flip_bit = (static_cast<uint32_t>(bit_dist(gen)) < 1);
          one_bits += flip_bit ? 1 : 0;
          word |= static_cast<uint32_t>(flip_bit) << j;
        }
        bits[i] = word;
      }

#pragma omp for
      for (size_t i = split_index; i < num_words; ++i) {
        uint32_t word = 0;
        for (size_t j = 0; j < 32; ++j) {
          bool const flip_bit = (static_cast<uint32_t>(bit_dist(gen)) < 99);
          one_bits += flip_bit ? 1 : 0;
          word |= static_cast<uint32_t>(flip_bit) << j;
        }
        bits[i] = word;
      }
    }
  }

  uint32_t* d_words_arr;
  gpuErrchk(cudaMalloc(&d_words_arr, num_words * sizeof(uint32_t)));
  gpuErrchk(cudaMemcpy(d_words_arr, bits.data(), num_words * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
  auto [blocks, threads] = getLaunchConfig(
      num_words / 32 == 0 ? 1 : num_words / 32, kMinTPB, kMaxTPB);
  for (uint8_t i = 0; i < num_levels; ++i) {
    writeWordsParallelKernel<<<blocks, threads>>>(ba, i, d_words_arr,
                                                  num_words);
  }
  kernelCheck();
  gpuErrchk(cudaFree(d_words_arr));

  if (one_bits_out != nullptr) {
    *one_bits_out = one_bits;
  }

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
                        size_t& max_memory_usage, uint32_t const GPU_index) {
  gpuErrchk(cudaSetDevice(GPU_index));

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
}  // namespace ecl