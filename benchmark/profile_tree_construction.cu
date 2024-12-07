#include <benchmark/benchmark.h>

#include <random>
#include <wavelet_tree.cuh>

namespace ecl {

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

}  // namespace ecl
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