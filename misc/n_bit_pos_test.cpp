#include <iostream>
#include <cstdint>
#include <cstring>
#include <bitset>
#include <bit>
#include <cassert>

#include <cmath>


template <uint32_t Value>
size_t getNBitPos(size_t n,uint32_t word) {
  static_assert(Value == 0 or Value == 1, "Template parameter must be 0 or 1");
  if constexpr (Value == 0) {
    // Find the position of the n-th zero in the word
    for (int i = 1; i < n; i++) {
      word = word | (word + 1);  // set least significant 0-bit
    }
    return ffs(~word) - 1;

  } else {
    // Find the position of the n-th one in the word
    for (int i = 1; i < n; i++) {
      word = word & (word - 1);  // clear least significant 1-bit
    }
    return ffs(word) - 1;
  }
}
 template <uint32_t Value, typename T>
  uint8_t getNBitPos(uint8_t const n, T word) {
    static_assert(Value == 0 or Value == 1,
                  "Template parameter must be 0 or 1");
    static_assert(std::is_integral<T>::value, "T must be an integral type.");
    static_assert(sizeof(T) == 4 or sizeof(T) == 8, "T must be 4 or 8 bytes.");
    assert(n > 0);
    if constexpr (sizeof(T) == 4) {
      assert(n <= (sizeof(uint32_t) * 8));
    if constexpr (Value == 0) {
      // Find the position of the n-th zero in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word | (word + 1);  // set least significant 0-bit
      }
      return ffs(~word) - 1;

    } else {
      // Find the position of the n-th one in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word & (word - 1);  // clear least significant 1-bit
      }
      return ffs(word) - 1;
    }
    } else {
      assert(n <= (sizeof(uint64_t) * 8));
      if constexpr (Value == 0) {
      // Find the position of the n-th zero in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word | (word + 1);  // set least significant 0-bit
      }
      return ffsll(~word) - 1;

    } else {
      // Find the position of the n-th one in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word & (word - 1);  // clear least significant 1-bit
      }
      return ffsll(word) - 1;
    }
    }
  }

  int main(int argc, char* argv[]) {
    uint64_t word = 1;  // 0x80000000
    //get n from command line
    size_t n_0 = atoi(argv[1]);
    size_t n_1 = atoi(argv[2]);
    size_t pos = getNBitPos<0, uint64_t>(n_0, word);
    size_t pos1 = getNBitPos<1, uint64_t>(n_1, word);
    // print binary representation of the word
    std::cout << "Word: " << std::bitset<64>(word) << std::endl;
    std::cout << "Position of " << n_0 << "-th zero: " << pos << std::endl;
    std::cout << "Position of " << n_1 << "-th one: " << pos1 << std::endl;
    return 0;
  }