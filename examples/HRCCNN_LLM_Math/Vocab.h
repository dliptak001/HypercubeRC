#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace hrccnn_llm_math {

// 20-token vocab: digits, operators, parens, '=', '.', ' ', '#'.
inline constexpr std::size_t kVocabSize = 20;
inline constexpr std::size_t kInputBits = 8;  // bipolar ASCII input channel count

/// Map a printable vocab character to its class index in [0, 20).
/// Returns -1 if the character is not in the vocab.
int CharToClass(char c);

/// Map a class index in [0, 20) back to its vocab character.
char ClassToChar(int cls);

/// Write 8 bipolar bits (+1 / -1) for the ASCII byte of c into out[0..7].
/// LSB first: out[0] = bit 0 of c, out[7] = bit 7.
void BipolarBits(char c, float out[kInputBits]);

/// Bulk-encode a string into row-major bipolar bits: out has size s.size()*8.
void BipolarEncode(const std::string& s, float* out);

}  // namespace hrccnn_llm_math
