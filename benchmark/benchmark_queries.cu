#include "ecl_wavelet/tree/wavelet_tree.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {

template <typename T>
size_t processAccessQueries(WaveletTree<T>& wt, std::vector<size_t>& queries,
                            size_t const num_iters) {
  std::vector<size_t> times(num_iters);
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  // warm-up
  for (size_t i = 0; i < 2; ++i) {
    auto results = wt.access(queries.data(), queries.size());
  }
  for (size_t i = 0; i < num_iters; ++i) {
    start = std::chrono::high_resolution_clock::now();
    auto results = wt.access(queries.data(), queries.size());
    end = std::chrono::high_resolution_clock::now();
    times[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }

  std::nth_element(times.begin(), times.begin() + times.size() / 2,
                   times.end());
  return times[times.size() / 2];
}
template <typename T, bool DoRank>
size_t processRSQueries(WaveletTree<T>& wt,
                        std::vector<RankSelectQuery<T>>& queries,
                        size_t const num_iters) {
  std::vector<size_t> times(num_iters);
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  // warm-up
  for (size_t i = 0; i < 2; ++i) {
    if constexpr (DoRank) {
      auto results = wt.rank(queries.data(), queries.size());
    } else {
      auto results = wt.select(queries.data(), queries.size());
    }
  }
  for (size_t i = 0; i < num_iters; ++i) {
    start = std::chrono::high_resolution_clock::now();
    if constexpr (DoRank) {
      auto results = wt.rank(queries.data(), queries.size());
    } else {
      auto results = wt.select(queries.data(), queries.size());
    }
    end = std::chrono::high_resolution_clock::now();
    times[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }

  std::nth_element(times.begin(), times.begin() + times.size() / 2,
                   times.end());
  return times[times.size() / 2];
}

template <typename T>
static void BM_queries(size_t const data_size, size_t const alphabet_size,
                       std::vector<size_t> const& num_queries_vec,
                       size_t const num_iters, uint32_t const GPU_index,
                       std::string const& output) {
  std::vector<T> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0ULL);
  auto const [data, hist] = generateRandomDataAndHist(alphabet, data_size);

  auto alphabet_copy = alphabet;
  WaveletTree<T> wt(data.data(), data_size, std::move(alphabet_copy),
                    GPU_index);

  for (auto const num_queries : num_queries_vec) {
    auto access_queries = generateRandomAccessQueries(data_size, num_queries);
    auto rank_queries =
        generateRandomRankQueries(data_size, num_queries, alphabet);
    auto select_queries =
        generateRandomSelectQueries(hist, num_queries, alphabet);

    size_t const median_access_time =
        processAccessQueries(wt, access_queries, num_iters);
    gpuErrchk(cudaHostRegister(access_queries.data(),
                               access_queries.size() * sizeof(size_t),
                               cudaHostRegisterPortable));
    size_t const median_access_time_pinned =
        processAccessQueries(wt, access_queries, num_iters);
    gpuErrchk(cudaHostUnregister(access_queries.data()));

    size_t const median_rank_time =
        processRSQueries<T, true>(wt, rank_queries, num_iters);
    size_t const median_select_time =
        processRSQueries<T, false>(wt, select_queries, num_iters);

    gpuErrchk(cudaHostRegister(rank_queries.data(),
                               rank_queries.size() * sizeof(RankSelectQuery<T>),
                               cudaHostRegisterPortable));
    gpuErrchk(
        cudaHostRegister(select_queries.data(),
                         select_queries.size() * sizeof(RankSelectQuery<T>),
                         cudaHostRegisterPortable));
    size_t const median_rank_time_pinned =
        processRSQueries<T, true>(wt, rank_queries, num_iters);
    size_t const median_select_time_pinned =
        processRSQueries<T, false>(wt, select_queries, num_iters);

    std::sort(
        rank_queries.begin(), rank_queries.end(),
        [](auto const& a, auto const& b) { return a.symbol_ < b.symbol_; });
    std::sort(
        select_queries.begin(), select_queries.end(),
        [](auto const& a, auto const& b) { return a.symbol_ < b.symbol_; });

    size_t const median_rank_time_sorted_pinned =
        processRSQueries<T, true>(wt, rank_queries, num_iters);
    size_t const median_select_time_sorted_pinned =
        processRSQueries<T, false>(wt, select_queries, num_iters);

    gpuErrchk(cudaHostUnregister(rank_queries.data()));
    gpuErrchk(cudaHostUnregister(select_queries.data()));

    size_t const median_rank_time_sorted =
        processRSQueries<T, true>(wt, rank_queries, num_iters);
    size_t const median_select_time_sorted =
        processRSQueries<T, false>(wt, select_queries, num_iters);
    std::ofstream file(output, std::ios_base::app);
    file << data_size << "," << alphabet_size << "," << num_queries << ","
         << median_access_time << "," << median_access_time_pinned << ","
         << median_rank_time << "," << median_select_time << ","
         << median_rank_time_sorted << "," << median_select_time_sorted << ","
         << median_rank_time_pinned << "," << median_select_time_pinned << ","
         << median_rank_time_sorted_pinned << ","
         << median_select_time_sorted_pinned << std::endl;
    file.close();
  }
}

}  // namespace ecl

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << "<num_iters> <output_dir> <GPU_index>"
              << std::endl;
    return EXIT_FAILURE;
  }

  uint32_t const num_iters = std::stoi(argv[1]);
  std::string const output_dir = argv[2];
  uint32_t const GPU_index = std::stoi(argv[3]);

  size_t const data_size = 6'000'000'000;

  std::vector<size_t> const num_queries_vec{100'000,   500'000,    1'000'000,
                                            5'000'000, 10'000'000, 100'000'000};

  std::vector<size_t> const alphabet_sizes{
      4,    6,    8,    12,    16,    24,    32,    48,    64,   96,
      128,  192,  256,  384,   512,   768,   1024,  1536,  2048, 3072,
      4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536};

  auto out_file = output_dir + "/ecl_queries_random_bm.csv";
  std::ofstream file(out_file);
  file << "data_size,alphabet_size,num_queries,median_access_time(mus),median_"
          "access_time_pinned,"
          "median_rank_time ,median_select_time,"
          "median_rank_time_sorted, "
          "median_select_time_sorted,median_rank_time_pinned,"
          "median_select_time_pinned,median_rank_time_sorted_pinned,"
          "median_select_time_sorted_pinned"
       << std::endl;
  file.close();

  for (auto const alphabet_size : alphabet_sizes) {
    if (alphabet_size <= 256) {
      ecl::BM_queries<uint8_t>(data_size, alphabet_size, num_queries_vec,
                               num_iters, GPU_index, out_file);
    } else {
      ecl::BM_queries<uint16_t>(data_size / 2, alphabet_size, num_queries_vec,
                                num_iters, GPU_index, out_file);
    }
  }
  return EXIT_SUCCESS;
}