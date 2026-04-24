#pragma once

/// @file Vocabulary.h
/// @brief Fixed 96-token ASCII vocabulary for character-level language modeling.
///
/// The Vocabulary maps between raw characters, integer class indices, and
/// bipolar bit vectors suitable for reservoir input.  The token set is fixed
/// at newline (0x0A) plus printable ASCII (0x20–0x7E) — 96 tokens total —
/// so trained models are portable across corpora without vocab mismatches.

#include <array>
#include <cstddef>
#include <string>

namespace lm {

class Vocabulary
{
public:
    static constexpr std::size_t kSize      = 96;
    static constexpr std::size_t kInputBits = 8;

    Vocabulary();

    /// Map a character to its class index, or -1 if not in the vocabulary.
    [[nodiscard]] int   CharToClass(char ch) const;

    /// Map a class index back to its character.  Returns '?' for out-of-range.
    [[nodiscard]] char  ClassToChar(int cls) const;

    /// True if the character is in the vocabulary.
    [[nodiscard]] bool  Contains(char ch) const;

    /// Encode a character as 8 bipolar bits (+1.0 / -1.0), LSB first.
    static void EncodeBipolar(char ch, float out[kInputBits]);

    /// Encode an entire string into a flat row-major bipolar array.
    /// @param out  Must have room for s.size() * kInputBits floats.
    static void EncodeBipolar(const std::string& s, float* out);

    /// The 96-character token string.  Position == class index.
    [[nodiscard]] const std::string& Tokens() const { return tokens_; }

private:
    static constexpr std::size_t kTableSize = 128;

    std::string                    tokens_;
    std::array<int, kTableSize>    char_to_class_;

    void BuildLookupTable();
};

}  // namespace lm
