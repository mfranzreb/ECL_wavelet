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

void tune_accessKernel(std::string out_dir, uint32_t const GPU_index) {
  uint8_t const num_iters = 100;
  auto const prop = getDeviceProperties();
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                  accessKernel<uint8_t, true, 1, true, true>));
  uint32_t max_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));
  size_t const data_size = prop.totalGlobalMem / 10;
  size_t num_warps =
      (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount + WS - 1) /
      WS;

  auto const GPU_name = prop.name;

  // Tune chunks vs queries
  std::string out_file = out_dir + "/access_chunks_vs_queries.csv";
  // Write column names to CSV
  std::ofstream file(out_file);
  file << "num_chunks,num_queries,time,GPU_name" << std::endl;
  file.close();

  std::vector<uint8_t> num_chunks_vec({2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
  std::vector<uint32_t> num_queries_vec({100'000, 500'000, 1'000'000, 5'000'000,
                                         10'000'000, 50'000'000, 100'000'000});
  size_t const alphabet_size = 4;
  auto [blocks, threads] = getLaunchConfig(num_warps, kMinTPB, max_size);

  std::vector<uint8_t> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto data = generateRandomData<uint8_t>(alphabet, data_size);

  WaveletTree<uint8_t> wt(data.data(), data_size, std::move(alphabet),
                          GPU_index);

  // create graphs for each number of chunks
  for (auto chunk : num_chunks_vec) {
    queries_graph_cache[chunk] = createQueriesGraph(chunk, 2);
  }

  std::chrono::high_resolution_clock::time_point start_time, end_time;
  IdealConfigs& ideal_configs = getIdealConfigs(GPU_name);
  for (uint32_t num_query : num_queries_vec) {
    auto queries =
        generateRandomAccessQueries(data_size, static_cast<size_t>(num_query));
    gpuErrchk(cudaHostRegister(queries.data(), num_query * sizeof(size_t),
                               cudaHostRegisterDefault));

    for (auto num_chunk : num_chunks_vec) {
      // Set ideal_configs slope so that correct num_chunks is chosen
      ideal_configs.accessKernel_logrel.slope =
          static_cast<float>(num_chunk) /
          std::log(static_cast<float>(num_query));
      // Warmup
      for (uint8_t i = 0; i < 2; ++i) {
        auto results = wt.template access<1>(queries.data(), num_query);
      }

      std::vector<float> times(num_iters);
      for (uint8_t i = 0; i < num_iters; ++i) {
        start_time = std::chrono::high_resolution_clock::now();
        auto results = wt.template access<1>(queries.data(), num_query);
        end_time = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time - start_time)
                       .count();
      }
      // Write median time to CSV
      std::nth_element(times.begin(), times.begin() + times.size() / 2,
                       times.end());
      std::ofstream file(out_file, std::ios_base::app);
      file << +num_chunk << "," << num_query << "," << times[num_iters / 2]
           << "," << GPU_name << std::endl;
      file.close();
    }
    gpuErrchk(cudaHostUnregister(queries.data()));
  }
  // Tune TPB and total warps
  size_t const num_queries =
      prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount * 100;
  out_file = out_dir + "/access_time_vs_warps.csv";
  // Write column names to CSV
  file.open(out_file);
  file << "num_warps,time,GPU_name" << std::endl;
  file.close();
  std::vector<uint32_t> num_warps_vec;
  // From 20% occupancy to 500%
  size_t const warps_at_20_occ = num_warps / 5;
  for (size_t i = warps_at_20_occ; i <= num_warps * 5; i += warps_at_20_occ) {
    num_warps_vec.push_back(i);
  }
  auto queries = generateRandomAccessQueries(data_size, num_queries);
  // register queries
  gpuErrchk(cudaHostRegister(queries.data(), num_queries * sizeof(size_t),
                             cudaHostRegisterDefault));

  auto const num_levels = ceilLog2Host(alphabet_size);
  for (auto num_warps : num_warps_vec) {
    auto const [blocks, threads] =
        getLaunchConfig(num_warps, max_size, max_size);
    if (blocks == -1 or threads == -1) {
      continue;
    }
    ideal_configs.ideal_tot_threads_accessKernel = blocks * threads;
    // Warmup
    for (uint8_t i = 0; i < 2; ++i) {
      auto results = wt.template access<1>(queries.data(), num_queries);
    }
    std::vector<size_t> times(num_iters);
    for (uint8_t i = 0; i < num_iters; ++i) {
      start_time = std::chrono::high_resolution_clock::now();
      auto results = wt.template access<1>(queries.data(), num_queries);
      end_time = std::chrono::high_resolution_clock::now();
      times[i] = std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                     .count();
    }
    // Write median time to CSV
    std::nth_element(times.begin(), times.begin() + times.size() / 2,
                     times.end());
    std::ofstream file(out_file, std::ios_base::app);
    file << num_warps << "," << times[times.size() / 2] << "," << GPU_name
         << std::endl;
    file.close();
  }
  gpuErrchk(cudaHostUnregister(queries.data()));
}

}  // namespace ecl

int main(int argc, char* argv[]) {
  auto const parent_dir = argv[1];
  auto const GPU_index = std::stoi(argv[2]);
  ecl::checkWarpSize(GPU_index);
  ecl::tune_accessKernel(std::string(parent_dir), GPU_index);
  return 0;
}