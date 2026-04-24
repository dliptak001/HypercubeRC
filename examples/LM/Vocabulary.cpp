#include "Vocabulary.h"

namespace lm {

Vocabulary::Vocabulary()
{
    tokens_.reserve(kSize);
    tokens_ += '\n';
    for (char c = 0x20; c <= 0x7E; ++c)
        tokens_ += c;

    BuildLookupTable();
}

void Vocabulary::BuildLookupTable()
{
    char_to_class_.fill(-1);
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
        auto byte = static_cast<unsigned char>(tokens_[i]);
        if (byte < kTableSize)
            char_to_class_[byte] = static_cast<int>(i);
    }
}

int Vocabulary::CharToClass(char ch) const
{
    auto byte = static_cast<unsigned char>(ch);
    return (byte < kTableSize) ? char_to_class_[byte] : -1;
}

char Vocabulary::ClassToChar(int cls) const
{
    if (cls < 0 || static_cast<std::size_t>(cls) >= tokens_.size())
        return '?';
    return tokens_[static_cast<std::size_t>(cls)];
}

bool Vocabulary::Contains(char ch) const
{
    return CharToClass(ch) >= 0;
}

void Vocabulary::EncodeBipolar(char ch, float out[kInputBits])
{
    auto byte = static_cast<unsigned char>(ch);
    for (std::size_t b = 0; b < kInputBits; ++b)
        out[b] = ((byte >> b) & 1u) ? 1.0f : -1.0f;
}

void Vocabulary::EncodeBipolar(const std::string& s, float* out)
{
    for (std::size_t i = 0; i < s.size(); ++i)
        EncodeBipolar(s[i], out + i * kInputBits);
}

}  // namespace lm
