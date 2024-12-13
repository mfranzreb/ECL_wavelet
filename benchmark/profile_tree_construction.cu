#include <random>
#include <wavelet_tree.cuh>

#include "test_benchmark_utils.cuh"

int main(int argc, char** argv) {
  // size is first command line argument
  auto const data_size = std::stoul(argv[1]);
  auto const alphabet_size = std::stoul(argv[2]);

  if (alphabet_size < std::numeric_limits<uint8_t>::max()) {
    auto [alphabet, data] =
        ecl::generateRandomAlphabetAndData<uint8_t>(alphabet_size, data_size);
    ecl::WaveletTree<uint8_t> wt(data.data(), data_size, std::move(alphabet),
                                 0);
  } else if (alphabet_size < std::numeric_limits<uint16_t>::max()) {
    auto [alphabet, data] =
        ecl::generateRandomAlphabetAndData<uint16_t>(alphabet_size, data_size);
    ecl::WaveletTree<uint16_t> wt(data.data(), data_size, std::move(alphabet),
                                  0);
  } else {
    auto [alphabet, data] =
        ecl::generateRandomAlphabetAndData<uint32_t>(alphabet_size, data_size);
    ecl::WaveletTree<uint32_t> wt(data.data(), data_size, std::move(alphabet),
                                  0);
  }
}