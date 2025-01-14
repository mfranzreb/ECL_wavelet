#include <bit_array.cuh>
#include <chrono>
#include <cstdint>
#include <rank_select.cuh>
#include <test_benchmark_utils.cuh>
#include <utils.cuh>
#include <vector>
#include <wavelet_tree.cuh>

namespace ecl {

void tune_accessKernel(std::string out_file, uint32_t const GPU_index) {
  using T = uint16_t;
  // Write column names to CSV
  std::ofstream file(out_file);
  file << "alphabet_size,num_blocks,num_threads,duration,use_shmem,GPU_name"
       << std::endl;

  std::vector<uint32_t> block_sizes;

  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, accessKernel<T, true>));
  uint32_t const max_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  for (uint32_t i = kMinTPB; i <= max_size; i *= 2) {
    block_sizes.push_back(i);
  }

  struct cudaDeviceProp prop = getDeviceProperties();
  size_t const data_size = prop.totalGlobalMem / 10;

  size_t const num_queries = 1'000'000;
  std::vector<size_t> queries(num_queries);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, data_size - 1);
  std::generate(queries.begin(), queries.end(), [&]() { return dis(gen); });

  size_t* d_queries;
  gpuErrchk(cudaMalloc(&d_queries, num_queries * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_queries, queries.data(), num_queries * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  T* d_results;
  gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(T)));

  uint8_t const alphabet_size = 4;
  std::vector<T> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto data = generateRandomData<T>(alphabet, data_size);
  WaveletTree<T> wt(data.data(), data_size, std::move(alphabet), GPU_index);
  uint32_t const min_warps = static_cast<uint32_t>(
      (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount + WS - 1) /
      WS);
  for (uint32_t num_warps = min_warps; num_warps <= num_queries;
       num_warps *= 2) {
    for (uint32_t block_size : block_sizes) {
      auto const [num_blocks, num_threads] =
          getLaunchConfig(num_warps, block_size, block_size);

      if (num_blocks == -1 or num_threads == -1) {
        continue;
      }
      bool const use_shmem =
          sizeof(size_t) * alphabet_size *
              (prop.maxThreadsPerMultiProcessor / num_threads) <=
          prop.sharedMemPerMultiprocessor;
      uint8_t const num_iters = 5;
      auto start = std::chrono::high_resolution_clock::now();
      for (uint8_t i = 0; i < num_iters; ++i) {
        if (use_shmem) {
          accessKernel<T, true>
              <<<num_blocks, num_threads, sizeof(size_t) * alphabet_size>>>(
                  wt, d_queries, num_queries, d_results);
        } else {
          accessKernel<T, false><<<num_blocks, num_threads>>>(
              wt, d_queries, num_queries, d_results);
        }
        kernelCheck();
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();

      // Write to CSV
      std::ofstream file(out_file, std::ios_base::app);
      file << alphabet_size << "," << num_blocks << "," << num_threads << ","
           << duration << "," << use_shmem << "," << prop.name << std::endl;
      file.close();
    }
  }
  gpuErrchk(cudaFree(d_queries));
  gpuErrchk(cudaFree(d_results));
}

void tune_rankKernel(std::string out_file, uint32_t const GPU_index) {
  using T = uint16_t;
  // Write column names to CSV
  std::ofstream file(out_file);
  file << "alphabet_size,num_blocks,num_threads,duration,use_shmem,GPU_name"
       << std::endl;

  std::vector<uint32_t> block_sizes;

  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, rankKernel<T, true>));
  uint32_t const max_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  for (uint32_t i = kMinTPB; i <= max_size; i *= 2) {
    block_sizes.push_back(i);
  }

  struct cudaDeviceProp prop = getDeviceProperties();
  size_t const data_size = prop.totalGlobalMem / 10;

  size_t const alphabet_size = 4;

  std::vector<T> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);

  size_t const num_queries = 1'000'000;
  std::vector<RankSelectQuery<T>> queries(num_queries);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, data_size - 1);
  std::uniform_int_distribution<T> dis2(0, alphabet.size() - 1);
  std::generate(queries.begin(), queries.end(), [&]() {
    return RankSelectQuery<T>(dis(gen), alphabet[dis2(gen)]);
  });
  RankSelectQuery<T>* d_queries;
  gpuErrchk(cudaMalloc(&d_queries, num_queries * sizeof(RankSelectQuery<T>)));

  gpuErrchk(cudaMemcpy(d_queries, queries.data(),
                       num_queries * sizeof(RankSelectQuery<T>),
                       cudaMemcpyHostToDevice));

  size_t* d_results;
  gpuErrchk(cudaMalloc(&d_results, num_queries * sizeof(size_t)));

  auto data = generateRandomData<T>(alphabet, data_size);
  WaveletTree<T> wt(data.data(), data_size, std::move(alphabet), GPU_index);
  uint32_t const min_warps = static_cast<uint32_t>(
      (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount + WS - 1) /
      WS);
  for (uint32_t num_warps = min_warps; num_warps <= num_queries;
       num_warps *= 2) {
    for (uint32_t block_size : block_sizes) {
      auto const [num_blocks, num_threads] =
          getLaunchConfig(num_warps, block_size, block_size);

      if (num_blocks == -1 or num_threads == -1) {
        continue;
      }
      bool const use_shmem =
          sizeof(size_t) * alphabet_size *
              (prop.maxThreadsPerMultiProcessor / num_threads) <=
          prop.sharedMemPerMultiprocessor;
      uint8_t const num_iters = 5;
      auto start = std::chrono::high_resolution_clock::now();
      for (uint8_t i = 0; i < num_iters; ++i) {
        if (use_shmem) {
          rankKernel<T, true>
              <<<num_blocks, num_threads, sizeof(size_t) * alphabet_size>>>(
                  wt, d_queries, num_queries, d_results);
        } else {
          rankKernel<T, false><<<num_blocks, num_threads>>>(
              wt, d_queries, num_queries, d_results);
        }
        kernelCheck();
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();

      // Write to CSV
      std::ofstream file(out_file, std::ios_base::app);
      file << alphabet_size << "," << num_blocks << "," << num_threads << ","
           << duration << "," << use_shmem << "," << prop.name << std::endl;
      file.close();
    }
  }
  gpuErrchk(cudaFree(d_queries));
  gpuErrchk(cudaFree(d_results));
}

void tune_calculateL2EntriesKernel(std::string out_file,
                                   uint32_t const GPU_index) {
  checkWarpSize(GPU_index);

  // Write column names to CSV
  std::ofstream file(out_file);
  file << "data_size,num_threads,duration,GPU_name" << std::endl;

  std::vector<uint32_t> block_sizes;

  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, calculateL2EntriesKernel));
  uint32_t const max_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  for (uint32_t i = kMinTPB; i <= max_size; i *= 2) {
    block_sizes.push_back(i);
  }

  struct cudaDeviceProp prop = getDeviceProperties();

  size_t const ba_size = prop.totalGlobalMem / 10;
  BitArray bit_array({ba_size});
  RankSelect rs(std::move(bit_array), GPU_index);
  size_t const num_blocks = (ba_size + RankSelectConfig::L1_BIT_SIZE - 1) /
                            RankSelectConfig::L1_BIT_SIZE;
  for (uint32_t block_size : block_sizes) {
    uint8_t const num_iters = 5;
    uint8_t const num_last_l2_blocks =
        (rs.bit_array_.sizeHost(0) % RankSelectConfig::L1_BIT_SIZE +
         RankSelectConfig::L2_BIT_SIZE - 1) /
        RankSelectConfig::L2_BIT_SIZE;
    auto start = std::chrono::high_resolution_clock::now();
    for (uint8_t i = 0; i < num_iters; ++i) {
      calculateL2EntriesKernel<<<num_blocks, block_size>>>(rs, 0,
                                                           num_last_l2_blocks);
      kernelCheck();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    // Write to CSV
    std::ofstream file(out_file, std::ios_base::app);
    file << ba_size << "," << block_size << "," << duration << "," << prop.name
         << std::endl;
  }
}

void tune_computeGlobalHistogramKernel(std::string out_file,
                                       uint32_t const GPU_index) {
  using T = uint8_t;
  // Write column names to CSV
  std::ofstream file(out_file);
  file << "data_size,alphabet_size,num_blocks,num_threads,duration,used_"
          "shmem,"
          "GPU_name"
       << std::endl;
  // Tuning params
  struct cudaDeviceProp prop = getDeviceProperties();
  std::string const GPU_name = prop.name;
  size_t const data_size = prop.totalGlobalMem / 2;
  T const alphabet_size = 4;
  struct cudaFuncAttributes funcAttrib;
  // No encoding in test, minimal alphabet
  gpuErrchk(cudaFuncGetAttributes(
      &funcAttrib, computeGlobalHistogramKernel<T, true, true, true>));
  // Using biggest size possible, since it allows for a higher bound on
  // shmem usage
  uint32_t const block_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  // Test also with low occupancy
  size_t min_warps = static_cast<size_t>((prop.maxThreadsPerMultiProcessor *
                                              prop.multiProcessorCount +
                                          WS - 1) /
                                         WS) /
                     5;

  // Tuning
  std::vector<T> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);

  auto alphabet_copy = alphabet;
  WaveletTree<T> wt(alphabet.data(), alphabet_size, std::move(alphabet_copy),
                    GPU_index);

  T* d_alphabet;
  gpuErrchk(cudaMalloc(&d_alphabet, alphabet_size * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_alphabet, alphabet.data(), alphabet_size * sizeof(T),
                       cudaMemcpyHostToDevice));

  size_t* d_histogram;
  gpuErrchk(cudaMalloc(&d_histogram, alphabet_size * sizeof(size_t)));

  auto data = generateRandomData<T>(alphabet, data_size);

  T* d_data;
  gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_data, data.data(), data_size * sizeof(T),
                       cudaMemcpyHostToDevice));

  size_t const max_warps = (data_size + WS - 1) / WS;
  for (size_t num_warps = min_warps; num_warps <= max_warps; num_warps *= 2) {
    auto const [num_blocks, num_threads] =
        getLaunchConfig(num_warps, block_size, block_size);

    if (num_blocks == -1 or num_threads == -1) {
      continue;
    }

    auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
    auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
    size_t const hist_size = sizeof(size_t) * alphabet_size;

    uint16_t const blocks_per_SM = max_threads_per_SM / num_threads;

    size_t const used_shmem =
        std::min(max_shmem_per_SM / blocks_per_SM, prop.sharedMemPerBlock);

    uint16_t const hists_per_block =
        std::min(static_cast<size_t>(num_threads), used_shmem / hist_size);

    uint8_t const num_iters = 5;
    auto start = std::chrono::high_resolution_clock::now();
    for (uint8_t i = 0; i < num_iters; ++i) {
      gpuErrchk(cudaMemset(d_histogram, 0, alphabet_size * sizeof(size_t)));
      if (hists_per_block > 0) {
        computeGlobalHistogramKernel<T, true, true, true>
            <<<num_blocks, num_threads, used_shmem>>>(
                wt, d_data, data_size, d_histogram, d_alphabet, alphabet_size,
                hists_per_block);
      } else {
        computeGlobalHistogramKernel<T, true, true, false>
            <<<num_blocks, num_threads>>>(wt, d_data, data_size, d_histogram,
                                          d_alphabet, alphabet_size,
                                          hists_per_block);
      }
      kernelCheck();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() /
        num_iters;

    // Write to CSV
    std::ofstream file(out_file, std::ios_base::app);
    file << data_size << "," << alphabet_size << "," << num_blocks << ","
         << num_threads << "," << duration << ","
         << int(1 ? hists_per_block > 0 : 0) << "," << GPU_name << std::endl;
  }
  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_alphabet));
  gpuErrchk(cudaFree(d_histogram));
}

void tune_fillLevelKernel(std::string out_file) {
  using T = uint8_t;

  // Write column names to CSV
  std::ofstream file(out_file);
  file << "data_size,num_blocks,num_threads,shmem_per_thread,"
          "duration,GPU_name"
       << std::endl;
  // Tuning params
  std::vector<uint32_t> block_sizes;

  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, fillLevelKernel<T, true>));
  uint32_t max_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, fillLevelKernel<T, false>));
  max_size =
      std::min(max_size, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  for (uint32_t i = kMinTPB; i <= max_size; i *= 2) {
    block_sizes.push_back(i);
  }

  struct cudaDeviceProp prop = getDeviceProperties();
  std::string const GPU_name = prop.name;
  cudaFuncSetAttribute(fillLevelKernel<T, true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       prop.sharedMemPerBlockOptin);

  size_t const min_warps = static_cast<size_t>(
      (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount + WS - 1) /
      WS);

  size_t const data_size = prop.totalGlobalMem / (4 * sizeof(T));

  auto const max_shmem_per_SM = prop.sharedMemPerMultiprocessor;
  auto const max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
  auto shmem_per_thread = sizeof(uint32_t) * 8 * sizeof(T);

  // Pad shmem banks if memory suffices
  size_t padded_shmem = std::numeric_limits<size_t>::max();
  if constexpr (kBankSizeBytes <= sizeof(T)) {
    // If the bank size is smaller than or equal to T, one element of
    // padding per thread is needed.
    padded_shmem =
        shmem_per_thread * max_threads_per_SM + sizeof(T) * max_threads_per_SM;
  } else {
    padded_shmem = shmem_per_thread * max_threads_per_SM +
                   shmem_per_thread * max_threads_per_SM / kBanksPerLine;
  }
  bool const enough_shmem = shmem_per_thread * padded_shmem <= max_shmem_per_SM;
  if (enough_shmem) {
    shmem_per_thread = padded_shmem / max_threads_per_SM;
  }

  // Tuning
  BitArray bit_array({data_size}, false);
  auto const data = generateRandomData<T>({0, 1, 2, 3}, data_size);
  for (uint32_t block_size : block_sizes) {
    size_t const max_warps = (data_size + WS - 1) / WS;
    for (size_t num_warps = min_warps; num_warps <= max_warps; num_warps *= 2) {
      T* d_data;
      gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
      gpuErrchk(cudaMemcpy(d_data, data.data(), data_size * sizeof(T),
                           cudaMemcpyHostToDevice));

      auto const [num_blocks, num_threads] =
          getLaunchConfig(num_warps, block_size, block_size);

      if (num_blocks == -1 or num_threads == -1) {
        continue;
      }

      auto start = std::chrono::high_resolution_clock::now();
      uint8_t const num_iters = 5;
      for (uint8_t i = 0; i < num_iters; ++i) {
        if (enough_shmem) {
          fillLevelKernel<T, true>
              <<<num_blocks, num_threads, shmem_per_thread * num_threads>>>(
                  bit_array, d_data, data_size, 1, 0);
        } else {
          fillLevelKernel<T, false><<<num_blocks, num_threads,
                                      sizeof(uint32_t) * (num_threads / WS)>>>(
              bit_array, d_data, data_size, 1, 0);
        }
        kernelCheck();
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() /
          num_iters;

      // Write to CSV
      std::ofstream file(out_file, std::ios_base::app);
      file << data_size << "," << num_blocks << "," << num_threads << ","
           << enough_shmem << "," << duration << "," << GPU_name << std::endl;
      gpuErrchk(cudaFree(d_data));
    }
  }
}

}  // namespace ecl

int main(int argc, char* argv[]) {
  auto const parent_dir = argv[1];
  auto const GPU_index = std::stoi(argv[2]);
  ecl::checkWarpSize(GPU_index);
  ecl::tune_accessKernel(std::string(parent_dir) + "/accessKernel.csv",
                         GPU_index);
  ecl::tune_rankKernel(std::string(parent_dir) + "/rankKernel.csv", GPU_index);
  ecl::tune_calculateL2EntriesKernel(
      std::string(parent_dir) + "/calculateL2EntriesKernel.csv", GPU_index);
  ecl::tune_fillLevelKernel(std::string(parent_dir) + "/fillLevelKernel.csv");
  ecl::tune_computeGlobalHistogramKernel(
      std::string(parent_dir) + "/computeGlobalHistogramKernel.csv", GPU_index);
  return 0;
}