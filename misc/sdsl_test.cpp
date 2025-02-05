#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include "sdsl/wavelet_trees.hpp"
#include "sdsl/wt_blcd.hpp"

std::vector<size_t> generateRandomQueries(size_t const data_size,
                                          size_t const num_queries) {
  std::vector<size_t> queries(num_queries);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, data_size - 1);
  std::generate(queries.begin(), queries.end(), [&]() { return dis(gen); });

  return queries;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> generateRandomAlphabetAndData(
    size_t const alphabet_size, size_t const data_size,
    bool enforce_alphabet_size = false) {
  if (alphabet_size < 3) {
    throw std::invalid_argument("Alphabet size must be at least 3");
  }
  std::vector<T> alphabet(alphabet_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
  size_t filled = 0;
  do {
    std::generate(alphabet.begin() + filled, alphabet.end(),
                  [&]() { return dis(gen); });
    // Check that all elements are unique
    std::sort(alphabet.begin(), alphabet.end());
    // remove duplicates
    auto it = std::unique(alphabet.begin(), alphabet.end());
    filled = std::distance(alphabet.begin(), it);
    if (not enforce_alphabet_size and filled > 2) {
      if (filled < alphabet_size) {
        alphabet.resize(filled);
      }
      break;
    }
  } while (filled != alphabet_size);

  std::vector<T> data(data_size);
  std::uniform_int_distribution<size_t> dis2(0, filled - 1);
  std::generate(data.begin(), data.end(),
                [&]() { return alphabet[dis2(gen)]; });

  return std::make_pair(alphabet, data);
}

int main(int argc, char *argv[]) {
  size_t data_size = std::stoull(argv[1]);
    size_t alphabet_size = std::stoull(argv[2]);
    size_t num_queries = std::stoull(argv[3]);

    auto queries = generateRandomQueries(data_size, num_queries);

    using T = uint16_t;

    auto [alphabet, data] =
      generateRandomAlphabetAndData<T>(alphabet_size, data_size, true);

      //Check that data is only formed of elements in the alphabet
      for (auto const &d : data) {
        assert(std::find(alphabet.begin(), alphabet.end(), d) != alphabet.end());
      }

// write data to file
  std::ofstream data_file("data_file");
  data_file.write(reinterpret_cast<const char*>(data.data()),
                  data.size() * sizeof(T));
  data_file.close();
  
  if constexpr (sizeof(T) == 1) {
    sdsl::wt_pc<sdsl::balanced_shape,
      sdsl::bit_vector,
      sdsl::rank_support_v5<>> wt;
    sdsl::construct(wt, "data_file", sizeof(T));
    // delete file
    std::remove("data_file");

    for (auto const i : queries) {
      assert(wt[i] == data[i]);
    }
  } else {
    sdsl::wt_pc<sdsl::balanced_shape,
      sdsl::bit_vector,
      sdsl::rank_support_v5<>,
      sdsl::wt_pc<sdsl::balanced_shape>::select_1_type,
      sdsl::wt_pc<sdsl::balanced_shape>::select_0_type,
      sdsl::int_tree<>> wt;
    sdsl::construct(wt, "data_file", sizeof(T));

    // delete file
    std::remove("data_file");

    for (auto const i : queries) {
      assert(wt[i] == data[i]);
    }
  }
}