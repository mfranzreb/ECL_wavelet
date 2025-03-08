#include <benchmark/benchmark.h>
#include <nvbio/basic/packed_vector.h>
#include <nvbio/strings/alphabet.h>
#include <nvbio/strings/wavelet_tree.h>

#include <random>
#include <utils.cuh>

#include "test_benchmark_utils.cuh"

namespace ecl {

template <nvbio::Alphabet AlphabetType>
nvbio::PackedVector<nvbio::device_tag,
                    nvbio::AlphabetTraits<AlphabetType>::SYMBOL_SIZE, true>
getNVbioArgs(size_t const data_size, uint8_t const alphabet_size = 0) {
  std::vector<uint8_t> alphabet;
  std::vector<uint8_t> data(data_size);
  if constexpr (AlphabetType == nvbio::DNA) {
    alphabet = std::vector<uint8_t>{'A', 'C', 'G', 'T'};
    data = generateRandomData<uint8_t>(alphabet, data_size);
  } else if constexpr (AlphabetType == nvbio::DNA_N) {
    alphabet = std::vector<uint8_t>{'A', 'C', 'G', 'T', 'N'};
    data = generateRandomData<uint8_t>(alphabet, data_size);
  } else if constexpr (AlphabetType == nvbio::PROTEIN) {
    alphabet = std::vector<uint8_t>{'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                    'S', 'T', 'V', 'W', 'Y', 'B', 'Z', 'X'};
    data = generateRandomData<uint8_t>(alphabet, data_size);
  } else if constexpr (AlphabetType == nvbio::ASCII) {
    std::tie(alphabet, data) =
        generateRandomAlphabetAndData<uint8_t>(alphabet_size, data_size, true);
  }
  uint32_t const alphabet_bits =
      nvbio::AlphabetTraits<AlphabetType>::SYMBOL_SIZE;

  // allocate a host packed vector
  nvbio::PackedVector<nvbio::host_tag, alphabet_bits, true> h_data(data_size);

  // pack the string
  nvbio::from_string<AlphabetType>(
      reinterpret_cast<const char*>(data.data()),
      reinterpret_cast<const char*>(data.data() + data.size()), h_data.begin());

  // copy it to the device
  nvbio::PackedVector<nvbio::device_tag, alphabet_bits, true> d_data(h_data);

  return d_data;
}

static void BM_NVBIO(benchmark::State& state) {
  if (state.range(0) > std::numeric_limits<uint32_t>::max()) {
    state.SkipWithError("Data size is too large for NVBIO.");
    return;
  }
  uint32_t const data_size = static_cast<uint32_t>(state.range(0));
  uint8_t const alphabet_size = static_cast<uint8_t>(state.range(1));

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;

  if (alphabet_size == 4) {
    auto d_data = getNVbioArgs<nvbio::DNA>(data_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    // Memory usage
    {
      size_t max_memory_usage = 0;
      std::atomic_bool done{false};
      std::atomic_bool can_start{false};
      std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                    std::ref(max_memory_usage), 0);
      while (not can_start) {
        std::this_thread::yield();
      }
      nvbio::WaveletTreeStorage<nvbio::device_tag> wt_mem;
      nvbio::setup(data_size, d_data.begin(), wt_mem);
      done = true;
      t.join();
      state.counters["memory_usage"] = max_memory_usage;
    }
    for (auto _ : state) {
      nvbio::setup(data_size, d_data.begin(), wt);
    }
  } else if (alphabet_size == 5) {
    auto d_data = getNVbioArgs<nvbio::DNA_N>(data_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    // Memory usage
    {
      size_t max_memory_usage = 0;
      std::atomic_bool done{false};
      std::atomic_bool can_start{false};
      std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                    std::ref(max_memory_usage), 0);
      while (not can_start) {
        std::this_thread::yield();
      }
      nvbio::WaveletTreeStorage<nvbio::device_tag> wt_mem;
      nvbio::setup(data_size, d_data.begin(), wt_mem);
      done = true;
      t.join();
      state.counters["memory_usage"] = max_memory_usage;
    }
    for (auto _ : state) {
      nvbio::setup(data_size, d_data.begin(), wt);
    }
  } else if (alphabet_size == 24) {
    auto d_data = getNVbioArgs<nvbio::PROTEIN>(data_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    // Memory usage
    {
      size_t max_memory_usage = 0;
      std::atomic_bool done{false};
      std::atomic_bool can_start{false};
      std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                    std::ref(max_memory_usage), 0);
      while (not can_start) {
        std::this_thread::yield();
      }
      nvbio::WaveletTreeStorage<nvbio::device_tag> wt_mem;
      nvbio::setup(data_size, d_data.begin(), wt_mem);
      done = true;
      t.join();
      state.counters["memory_usage"] = max_memory_usage;
    }
    for (auto _ : state) {
      nvbio::setup(data_size, d_data.begin(), wt);
    }
  } else {
    auto d_data = getNVbioArgs<nvbio::ASCII>(data_size, alphabet_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    // Memory usage
    {
      size_t max_memory_usage = 0;
      std::atomic_bool done{false};
      std::atomic_bool can_start{false};
      std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                    std::ref(max_memory_usage), 0);
      while (not can_start) {
        std::this_thread::yield();
      }
      nvbio::WaveletTreeStorage<nvbio::device_tag> wt_mem;
      nvbio::setup(data_size, d_data.begin(), wt_mem);
      done = true;
      t.join();
      state.counters["memory_usage"] = max_memory_usage;
    }
    for (auto _ : state) {
      nvbio::setup(data_size, d_data.begin(), wt);
    }
  }
}

// For initializing CUDA
BENCHMARK(BM_NVBIO)
    ->ArgsProduct({{200'000'000}, {4}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_NVBIO)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'500'000'000, 2'000'000'000},
                   {4, 5, 24, 64, 100, 155, 256}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl