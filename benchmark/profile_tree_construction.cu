#include <random>

#include "ecl_wavelet/tree/wavelet_tree.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"

int main([[maybe_unused]] int argc, char** argv) {
  // size is first command line argument
  auto const data_size = std::stoul(argv[1]);
  auto const alphabet_size = std::stoul(argv[2]);
  auto const use_min_alphabet = static_cast<bool>(std::stoi(argv[3]));

  if (alphabet_size < std::numeric_limits<uint8_t>::max()) {
    std::vector<uint8_t> alphabet;
    std::vector<uint8_t> data;
    if (use_min_alphabet) {
      alphabet = std::vector<uint8_t>(alphabet_size);
      std::iota(alphabet.begin(), alphabet.end(), 0ULL);
      data = ecl::utils::generateRandomData<uint8_t>(alphabet, data_size);
    } else {
      std::tie(alphabet, data) =
          ecl::utils::generateRandomAlphabetAndData<uint8_t>(alphabet_size,
                                                             data_size, true);
    }
    ecl::WaveletTree<uint8_t> wt(data.data(), data_size, std::move(alphabet),
                                 0);
  } else if (alphabet_size < std::numeric_limits<uint16_t>::max()) {
    std::vector<uint16_t> alphabet;
    std::vector<uint16_t> data;
    if (use_min_alphabet) {
      alphabet = std::vector<uint16_t>(alphabet_size);
      std::iota(alphabet.begin(), alphabet.end(), 0ULL);
      data = ecl::utils::generateRandomData<uint16_t>(alphabet, data_size);
    } else {
      std::tie(alphabet, data) =
          ecl::utils::generateRandomAlphabetAndData<uint16_t>(alphabet_size,
                                                              data_size, true);
    }
    ecl::WaveletTree<uint16_t> wt(data.data(), data_size, std::move(alphabet),
                                  0);
  } else {
    std::vector<uint32_t> alphabet;
    std::vector<uint32_t> data;
    if (use_min_alphabet) {
      alphabet = std::vector<uint32_t>(alphabet_size);
      std::iota(alphabet.begin(), alphabet.end(), 0ULL);
      data = ecl::utils::generateRandomData<uint32_t>(alphabet, data_size);
    } else {
      std::tie(alphabet, data) =
          ecl::utils::generateRandomAlphabetAndData<uint32_t>(alphabet_size,
                                                              data_size, true);
    }
    ecl::WaveletTree<uint32_t> wt(data.data(), data_size, std::move(alphabet),
                                  0);
  }
}