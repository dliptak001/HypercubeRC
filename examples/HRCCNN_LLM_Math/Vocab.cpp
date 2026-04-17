#include "Vocab.h"

#include <array>

namespace hrccnn_llm_math {

namespace {

// Class index -> character. Order is load-bearing only for stability of
// class indices across runs (and across train/eval); the ordering itself
// is arbitrary.
constexpr std::array<char, kVocabSize> kClassToChar = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '-', '*', '/',
    '(', ')',
    '=', '.', ' ', '#'
};

}  // namespace

int CharToClass(char c)
{
    for (std::size_t i = 0; i < kVocabSize; ++i) {
        if (kClassToChar[i] == c) return static_cast<int>(i);
    }
    return -1;
}

char ClassToChar(int cls)
{
    if (cls < 0 || static_cast<std::size_t>(cls) >= kVocabSize) return '?';
    return kClassToChar[cls];
}

void BipolarBits(char c, float out[kInputBits])
{
    unsigned char byte = static_cast<unsigned char>(c);
    for (std::size_t b = 0; b < kInputBits; ++b) {
        out[b] = ((byte >> b) & 1u) ? 1.0f : -1.0f;
    }
}

void BipolarEncode(const std::string& s, float* out)
{
    for (std::size_t i = 0; i < s.size(); ++i) {
        BipolarBits(s[i], out + i * kInputBits);
    }
}

}  // namespace hrccnn_llm_math
