#include <omp.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
std::vector<T> generateRandomData(std::vector<T> const& alphabet,
                                  size_t const data_size) {
  std::vector<T> data(data_size);
#pragma omp parallel
  {
    // Create a thread-local random number generator
    std::random_device rd;
    std::mt19937 gen(rd() + omp_get_thread_num());  // Add thread number to seed
                                                    // for better randomness
    std::uniform_int_distribution<size_t> dis(0, alphabet.size() - 1);

#pragma omp for
    for (size_t i = 0; i < data_size; i++) {
      data[i] = alphabet[dis(gen)];
    }
  }

  return data;
}

int main(int argc, char* argv[]) {
  // first argument is the size of the data, second is the size of the alphabet
  // and third the data file
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <data_size> <alphabet_size> <data_file>" << std::endl;
    return 1;
  }
  size_t const data_size = std::stoul(argv[1]);
  size_t const alphabet_size = std::stoul(argv[2]);
  std::string const data_file = argv[3];
  if (alphabet_size <= 256) {
    std::vector<uint8_t> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    auto const data = generateRandomData(alphabet, data_size);
    std::ofstream file(data_file, std::ios::binary);
    file.write(reinterpret_cast<char const*>(data.data()), data_size);
  } else {
    std::vector<uint16_t> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    auto const data = generateRandomData(alphabet, data_size);
    std::ofstream file(data_file, std::ios::binary);
    file.write(reinterpret_cast<char const*>(data.data()),
               data_size * sizeof(uint16_t));
  }
  return 0;
}