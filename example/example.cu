#include <ecl_wavelet/tree/wavelet_tree.cuh>
#include <random>
#include <span>
#include <vector>

int main() {
  // Generate a random DNA sequence
  std::vector<uint8_t> dna_alphabet = {0, 1, 2, 3};  // A, C, G, T
  std::vector<uint8_t> dna_sequence(1000);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis(0, dna_alphabet.size() - 1);
  for (size_t i = 0; i < dna_sequence.size(); ++i) {
    dna_sequence[i] = dna_alphabet[dis(gen)];
  }

  // Create a Wavelet Tree
  ecl::WaveletTree<uint8_t> wavelet_tree;
  try {
    // Pass the alphabet only if you're certain it's correct
    wavelet_tree = ecl::WaveletTree<uint8_t>(
        dna_sequence.data(), dna_sequence.size(), std::vector<uint8_t>(), 0);
  } catch (const std::runtime_error& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }

  // Access all the elements in the tree
  std::vector<size_t> access_queries(dna_sequence.size());
  std::iota(access_queries.begin(), access_queries.end(), 0);
  try {
    std::span<uint8_t> access_results =
        wavelet_tree.access(access_queries.data(), access_queries.size());
  } catch (const std::runtime_error& e) {
    // Could process them in smaller batches to avoid memory issues
    std::cerr << e.what() << '\n';
    return 1;
  }

  // Find out the occurrences of each symbol up to the middle of the sequence
  std::vector<ecl::RankSelectQuery<uint8_t>> rank_queries{
      {dna_sequence.size() / 2, 0},
      {dna_sequence.size() / 2, 1},
      {dna_sequence.size() / 2, 2},
      {dna_sequence.size() / 2, 3}};

  try {
    std::span<size_t> rank_results =
        wavelet_tree.rank(rank_queries.data(), rank_queries.size());
    for (size_t i = 0; i < rank_queries.size(); ++i) {
      std::cout << "Rank of symbol " << +rank_queries[i].symbol_
                << " up to index " << rank_queries[i].index_ << ": "
                << rank_results[i] << std::endl;
    }
  } catch (const std::runtime_error& e) {
    // Could process them in smaller batches to avoid memory issues
    std::cerr << e.what() << '\n';
    return 1;
  }

  // Find the position of the first and last occurrence of each symbol
  std::vector<ecl::RankSelectQuery<uint8_t>> select_queries{
      {1, 0},
      {1, 1},
      {1, 2},
      {1, 3},
      {wavelet_tree.getTotalAppearances(0), 0},
      {wavelet_tree.getTotalAppearances(1), 1},
      {wavelet_tree.getTotalAppearances(2), 2},
      {wavelet_tree.getTotalAppearances(3), 3}};

  try {
    std::span<size_t> select_results =
        wavelet_tree.select(select_queries.data(), select_queries.size());
    for (size_t i = 0; i < select_queries.size(); ++i) {
      std::cout << "Position of " << select_queries[i].index_
                << "th occurrence of " << +select_queries[i].symbol_ << ": "
                << select_results[i] << std::endl;
    }
  } catch (const std::runtime_error& e) {
    // Could process them in smaller batches to avoid memory issues
    std::cerr << e.what() << '\n';
    return 1;
  }

  return 0;
}