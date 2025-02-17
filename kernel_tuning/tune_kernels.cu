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

void tuneQueries(std::string out_dir, uint32_t const GPU_index) {
  uint8_t const num_iters = 1;
  auto const prop = getDeviceProperties();
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
        auto results = wt.template access<1>(access_queries.data(), num_query);
      }

      std::vector<size_t> times(num_iters);
      for (uint8_t i = 0; i < num_iters; ++i) {
        start_time = std::chrono::high_resolution_clock::now();
        auto results = wt.template access<1>(access_queries.data(), num_query);
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
        auto results = wt.template rank<1>(rank_queries.data(), num_query);
      }

      for (uint8_t i = 0; i < num_iters; ++i) {
        start_time = std::chrono::high_resolution_clock::now();
        auto results = wt.template rank<1>(rank_queries.data(), num_query);
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
        auto results = wt.template select<1>(select_queries.data(), num_query);
      }

      for (uint8_t i = 0; i < num_iters; ++i) {
        start_time = std::chrono::high_resolution_clock::now();
        auto results = wt.template select<1>(select_queries.data(), num_query);
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
}  // namespace ecl

int main(int argc, char* argv[]) {
  auto const parent_dir = argv[1];
  auto const GPU_index = std::stoi(argv[2]);
  ecl::checkWarpSize(GPU_index);
  ecl::tuneQueries(std::string(parent_dir), GPU_index);
  return 0;
}