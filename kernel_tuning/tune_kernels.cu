#include <algorithm>
#include <bit_array.cuh>
#include <chrono>
#include <cstdint>
#include <rank_select.cuh>
#include <test_benchmark_utils.cuh>
#include <utils.cuh>
#include <vector>
#include <wavelet_tree.cuh>

namespace ecl {

template <typename T>
class WaveletTreeTest : public WaveletTree<T> {
 public:
  using WaveletTree<T>::WaveletTree;
  using WaveletTree<T>::computeGlobalHistogram;
  using WaveletTree<T>::fillLevel;
  using WaveletTree<T>::alphabet_;
};

void tuneQueries(std::string out_dir, uint32_t const GPU_index) {
  uint8_t const num_iters = 100;
  auto const& prop = getDeviceProperties();
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                  accessKernel<uint8_t, true, 1, true, true>));
  uint32_t max_size_access =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                  rankKernel<uint8_t, true, 1, true, true>));
  uint32_t max_size_rank =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, selectKernel<uint8_t, 1, true>));
  uint32_t max_size_select =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  size_t const data_size = prop.totalGlobalMem / 10;

  auto const GPU_name = prop.name;

  // Tune chunks vs access_queries
  std::string out_file_access = out_dir + "/access_chunks_vs_queries.csv";
  std::string out_file_rank = out_dir + "/rank_chunks_vs_queries.csv";
  std::string out_file_select = out_dir + "/select_chunks_vs_queries.csv";
  // Write column names to CSV
  std::ofstream file(out_file_access);
  file << "num_chunks,num_queries,time,GPU_name" << std::endl;
  file.close();
  file = std::ofstream(out_file_rank);
  file << "num_chunks,num_queries,time,GPU_name" << std::endl;
  file.close();
  file = std::ofstream(out_file_select);
  file << "num_chunks,num_queries,time,GPU_name" << std::endl;
  file.close();

  std::vector<uint8_t> num_chunks_vec({2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
  std::vector<uint32_t> num_queries_vec({100'000, 500'000, 1'000'000, 5'000'000,
                                         10'000'000, 50'000'000, 100'000'000});
  size_t const alphabet_size = 16;

  std::vector<uint8_t> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto [data, hist] = generateRandomDataAndHist(alphabet, data_size);
  auto alphabet_copy = alphabet;

  WaveletTree<uint8_t> wt(data.data(), data_size, std::move(alphabet_copy),
                          GPU_index);

  // create graphs for each number of chunks
  for (auto chunk : num_chunks_vec) {
    queries_graph_cache[chunk] = createQueriesGraph(chunk, 2);
  }

  std::chrono::high_resolution_clock::time_point start_time, end_time;
  IdealConfigs& ideal_configs = getIdealConfigs(GPU_name);
  for (uint32_t num_query : num_queries_vec) {
    auto access_queries =
        generateRandomAccessQueries(data_size, static_cast<size_t>(num_query));
    auto rank_queries = generateRandomRankQueries(
        data_size, static_cast<size_t>(num_query), alphabet);
    auto select_queries = generateRandomSelectQueries(
        hist, static_cast<size_t>(num_query), alphabet);
    gpuErrchk(cudaHostRegister(access_queries.data(),
                               num_query * sizeof(size_t),
                               cudaHostRegisterDefault));
    gpuErrchk(cudaHostRegister(rank_queries.data(),
                               num_query * sizeof(RankSelectQuery<uint8_t>),
                               cudaHostRegisterDefault));
    gpuErrchk(cudaHostRegister(select_queries.data(),
                               num_query * sizeof(RankSelectQuery<uint8_t>),
                               cudaHostRegisterDefault));

    for (auto num_chunk : num_chunks_vec) {
      // Set ideal_configs slope so that correct num_chunks is chosen
      float const slope = static_cast<float>(num_chunk) /
                          std::log(static_cast<float>(num_query));
      ideal_configs.accessKernel_logrel.slope = slope;
      ideal_configs.rankKernel_logrel.slope = slope;
      ideal_configs.selectKernel_logrel.slope = slope;
      // Warmup
      for (uint8_t i = 0; i < 2; ++i) {
        auto results = wt.access(access_queries.data(), num_query);
      }

      std::vector<size_t> times(num_iters);
      for (uint8_t i = 0; i < num_iters; ++i) {
        start_time = std::chrono::high_resolution_clock::now();
        auto results = wt.access(access_queries.data(), num_query);
        end_time = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time - start_time)
                       .count();
      }
      // Write median time to CSV
      std::nth_element(times.begin(), times.begin() + times.size() / 2,
                       times.end());
      std::ofstream file(out_file_access, std::ios_base::app);
      file << +num_chunk << "," << num_query << "," << times[num_iters / 2]
           << "," << GPU_name << std::endl;
      file.close();

      // Warmup
      for (uint8_t i = 0; i < 2; ++i) {
        auto results = wt.rank(rank_queries.data(), num_query);
      }

      for (uint8_t i = 0; i < num_iters; ++i) {
        start_time = std::chrono::high_resolution_clock::now();
        auto results = wt.rank(rank_queries.data(), num_query);
        end_time = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time - start_time)
                       .count();
      }
      // Write median time to CSV
      std::nth_element(times.begin(), times.begin() + times.size() / 2,
                       times.end());
      file = std::ofstream(out_file_rank, std::ios_base::app);
      file << +num_chunk << "," << num_query << "," << times[num_iters / 2]
           << "," << GPU_name << std::endl;
      file.close();

      // Warmup
      for (uint8_t i = 0; i < 2; ++i) {
        auto results = wt.select(select_queries.data(), num_query);
      }

      for (uint8_t i = 0; i < num_iters; ++i) {
        start_time = std::chrono::high_resolution_clock::now();
        auto results = wt.select(select_queries.data(), num_query);
        end_time = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time - start_time)
                       .count();
      }
      // Write median time to CSV
      std::nth_element(times.begin(), times.begin() + times.size() / 2,
                       times.end());
      file = std::ofstream(out_file_select, std::ios_base::app);
      file << +num_chunk << "," << num_query << "," << times[num_iters / 2]
           << "," << GPU_name << std::endl;
      file.close();
    }
    gpuErrchk(cudaHostUnregister(access_queries.data()));
    gpuErrchk(cudaHostUnregister(rank_queries.data()));
    gpuErrchk(cudaHostUnregister(select_queries.data()));
  }
}

void tuneL2entriesKernel(std::string out_dir, uint32_t const GPU_index) {
  uint8_t const num_iters = 100;
  auto const& prop = getDeviceProperties();
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, calculateL2EntriesKernel));
  uint32_t max_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  size_t const data_size = prop.totalGlobalMem / 10;

  auto const GPU_name = prop.name;

  // Write column names to CSV
  std::string out_file = out_dir + "/calculateL2EntriesKernel_TPB.csv";
  std::ofstream file(out_file);
  file << "tpb,time,GPU_name" << std::endl;
  file.close();

  std::vector<size_t> threads_per_block_vec;
  for (size_t i = max_size; i >= kMinTPB; i /= 2) {
    threads_per_block_vec.push_back(i);
  }
  BitArray bit_array = createRandomBitArray(data_size, 1);
  RankSelect rs(std::move(bit_array), GPU_index);
  auto num_last_l2_blocks =
      (data_size % RSConfig::L1_BIT_SIZE + RSConfig::L2_BIT_SIZE - 1) /
      RSConfig::L2_BIT_SIZE;
  if (num_last_l2_blocks == 0) {
    num_last_l2_blocks = RSConfig::NUM_L2_PER_L1;
  }
  auto const num_l1_blocks =
      (data_size + RSConfig::L1_BIT_SIZE - 1) / RSConfig::L1_BIT_SIZE;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (auto const tpb : threads_per_block_vec) {
    // Warmup
    for (uint8_t i = 0; i < 2; ++i) {
      calculateL2EntriesKernel<<<num_l1_blocks, tpb>>>(
          rs, 0, num_last_l2_blocks, num_l1_blocks, tpb,
          (RSConfig::NUM_L2_PER_L1 + tpb - 1) / tpb, data_size);
      kernelCheck();
    }

    std::vector<float> times(num_iters);
    for (uint8_t i = 0; i < num_iters; ++i) {
      cudaEventRecord(start);
      calculateL2EntriesKernel<<<num_l1_blocks, tpb>>>(
          rs, 0, num_last_l2_blocks, num_l1_blocks, tpb,
          (RSConfig::NUM_L2_PER_L1 + tpb - 1) / tpb, data_size);
      cudaEventRecord(stop);
      kernelCheck();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&times[i], start, stop);
    }
    // Write median time to CSV
    std::nth_element(times.begin(), times.begin() + times.size() / 2,
                     times.end());
    std::ofstream file(out_file, std::ios_base::app);
    file << tpb << "," << times[num_iters / 2] << "," << GPU_name << std::endl;
    file.close();
  }
}

void tuneFillLevelKernel(std::string out_dir, uint32_t const GPU_index) {
  uint8_t const num_iters = 100;
  auto const& prop = getDeviceProperties();
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(
      cudaFuncGetAttributes(&funcAttrib, fillLevelKernel<uint16_t, true>));
  uint32_t max_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  size_t const data_size = prop.totalGlobalMem / 10;

  auto const GPU_name = prop.name;

  // Write column names to CSV
  std::string out_file_tpb = out_dir + "/fillLevelKernel_TPB.csv";
  std::ofstream file(out_file_tpb);
  file << "tpb,alphabet_size,time,GPU_name" << std::endl;
  file.close();

  std::vector<size_t> threads_per_block_vec;
  for (size_t i = max_size; i >= kMinTPB; i /= 2) {
    threads_per_block_vec.push_back(i);
  }

  size_t const alphabet_size = 4;
  IdealConfigs& ideal_configs = getIdealConfigs(GPU_name);

  BitArray bit_array(std::vector<size_t>{data_size}, false);

  using T = uint8_t;
  std::vector<T> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto data = generateRandomData(alphabet, data_size);
  WaveletTreeTest<T> wt(data.data(), data_size, std::move(alphabet), GPU_index);
  T* d_data;
  gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(T)));
  gpuErrchk(cudaMemcpy(d_data, data.data(), data_size * sizeof(T),
                       cudaMemcpyHostToDevice));
  for (auto const tpb : threads_per_block_vec) {
    ideal_configs.ideal_TPB_fillLevelKernel = tpb;
    // Warmup
    for (uint8_t i = 0; i < 2; ++i) {
      wt.fillLevel(bit_array, d_data, data_size, 0);
    }

    std::vector<size_t> times(num_iters);
    for (uint8_t i = 0; i < num_iters; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      wt.fillLevel(bit_array, d_data, data_size, 0);
      auto end = std::chrono::high_resolution_clock::now();
      times[i] =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
    }
    // Write median time to CSV
    std::nth_element(times.begin(), times.begin() + times.size() / 2,
                     times.end());
    std::ofstream file(out_file_tpb, std::ios_base::app);
    file << tpb << "," << alphabet_size << "," << times[num_iters / 2] << ","
         << GPU_name << std::endl;
    file.close();
  }
  gpuErrchk(cudaFree(d_data));
}

__global__ static void getTotalNumValsKernel(RankSelect rank_select,
                                             uint32_t array_index,
                                             size_t* output, bool const val) {
  if (val) {
    *output = rank_select.getTotalNumVals<1>(array_index);
  } else {
    *output = rank_select.getTotalNumVals<0>(array_index);
  }
}

void tuneSamplesKernel(std::string out_dir, uint32_t const GPU_index) {
  uint8_t const num_iters = 100;
  auto const& prop = getDeviceProperties();
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, calculateSelectSamplesKernel));
  uint32_t max_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  size_t const data_size = prop.totalGlobalMem / 10;

  auto const GPU_name = prop.name;

  // Write column names to CSV
  std::string out_file = out_dir + "/calculateSelectSamplesKernel_TPB.csv";
  std::ofstream file(out_file);
  file << "tpb,time,GPU_name" << std::endl;
  file.close();

  std::vector<size_t> threads_per_block_vec;
  for (size_t i = max_size; i >= kMinTPB; i /= 2) {
    threads_per_block_vec.push_back(i);
  }
  auto const num_warps =
      (prop.maxThreadsPerBlock * prop.multiProcessorCount + WS - 1) / WS;
  auto bit_array = createRandomBitArray(data_size, 1);
  RankSelect rs(std::move(bit_array), GPU_index);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  size_t* num_ones;
  gpuErrchk(cudaMallocManaged(&num_ones, sizeof(size_t)));
  getTotalNumValsKernel<<<1, 1>>>(rs, 0, num_ones, true);
  kernelCheck();
  size_t const num_ones_samples = *num_ones / RSConfig::SELECT_SAMPLE_RATE;
  size_t const num_zeros_samples =
      (data_size - *num_ones) / RSConfig::SELECT_SAMPLE_RATE;
  gpuErrchk(cudaFree(num_ones));
  for (auto const tpb : threads_per_block_vec) {
    auto const [num_blocks, block_size] = getLaunchConfig(num_warps, tpb, tpb);

    if (num_blocks == -1 or block_size == -1) {
      continue;
    }
    // Warmup
    for (uint8_t i = 0; i < 2; ++i) {
      calculateSelectSamplesKernel<<<num_blocks, block_size>>>(
          rs, 0, num_blocks * block_size, num_ones_samples, num_zeros_samples);
      kernelCheck();
    }

    std::vector<float> times(num_iters);
    for (uint8_t i = 0; i < num_iters; ++i) {
      cudaEventRecord(start);
      calculateSelectSamplesKernel<<<num_blocks, block_size>>>(
          rs, 0, num_blocks * block_size, num_ones_samples, num_zeros_samples);
      cudaEventRecord(stop);
      kernelCheck();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&times[i], start, stop);
    }
    // Write median time to CSV
    std::nth_element(times.begin(), times.begin() + times.size() / 2,
                     times.end());
    std::ofstream file(out_file, std::ios_base::app);
    file << tpb << "," << times[num_iters / 2] << "," << GPU_name << std::endl;
    file.close();
  }

  // tune tot_threads
  out_file = out_dir + "/calculateSelectSamplesKernel_tot_threads.csv";
  file = std::ofstream(out_file);
  file << "tot_threads,time,GPU_name" << std::endl;
  file.close();

  size_t const max_occ = prop.maxThreadsPerBlock * prop.multiProcessorCount;
  std::vector<size_t> tot_threads_vec{max_occ / 5, 2 * max_occ / 5,
                                      3 * max_occ / 5, 4 * max_occ / 5};
  // go from 100% to 500% occupancy
  for (size_t i = 1; i <= 5; ++i) {
    tot_threads_vec.push_back(max_occ * i);
  }
  for (auto const tot_threads : tot_threads_vec) {
    auto const [num_blocks, block_size] =
        getLaunchConfig(tot_threads / WS, max_size, max_size);
    if (num_blocks == -1 or block_size == -1) {
      continue;
    }
    // Warmup
    for (uint8_t i = 0; i < 2; ++i) {
      calculateSelectSamplesKernel<<<num_blocks, block_size>>>(
          rs, 0, num_blocks * block_size, num_ones_samples, num_zeros_samples);
      kernelCheck();
    }

    std::vector<float> times(num_iters);
    for (uint8_t i = 0; i < num_iters; ++i) {
      cudaEventRecord(start);
      calculateSelectSamplesKernel<<<num_blocks, block_size>>>(
          rs, 0, num_blocks * block_size, num_ones_samples, num_zeros_samples);
      cudaEventRecord(stop);
      kernelCheck();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&times[i], start, stop);
    }
    // Write median time to CSV
    std::nth_element(times.begin(), times.begin() + times.size() / 2,
                     times.end());
    std::ofstream file(out_file, std::ios_base::app);
    file << tot_threads << "," << times[num_iters / 2] << "," << GPU_name
         << std::endl;
    file.close();
  }
}

}  // namespace ecl

int main(int argc, char* argv[]) {
  auto const parent_dir = argv[1];
  auto const GPU_index = std::stoi(argv[2]);
  ecl::checkWarpSize(GPU_index);
  ecl::tuneQueries(std::string(parent_dir), GPU_index);
  ecl::tuneL2entriesKernel(std::string(parent_dir), GPU_index);
  ecl::tuneFillLevelKernel(std::string(parent_dir), GPU_index);
  ecl::tuneSamplesKernel(std::string(parent_dir), GPU_index);
  return 0;
}