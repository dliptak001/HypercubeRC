#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace hrccnn_lm_text {

inline constexpr std::size_t kInputBits  = 8;    ///< bipolar ASCII input width
inline constexpr std::size_t kVocabCap   = 128;  ///< upper bound on ASCII vocab

/// Text corpus + derived vocabulary.  `vocab` is the sorted unique-char
/// string from `text`; a char's class index is its position in `vocab`.
struct Corpus
{
    std::string text;
    std::string vocab;                  ///< position == class index
    std::array<int, kVocabCap> char_to_class{};  ///< ASCII lookup, -1 if not in vocab
};

/// Load a plain-text corpus from disk and derive the vocab.  Returns false
/// on missing file or empty corpus.  Populates `out.text`, `out.vocab`,
/// `out.char_to_class`.
bool LoadCorpus(const std::string& path, Corpus& out);

/// Build a Corpus over an already-loaded `text` string with a fixed vocab.
/// Used by eval/infer where the vocab comes from the saved model (so a
/// char that never appeared in training stays unmappable even if it
/// appears in a fresh corpus).
void AttachCorpus(const std::string& text, const std::string& vocab, Corpus& out);

/// Character -> class index ([0, vocab.size())), or -1 if not in vocab.
inline int CharToClass(const Corpus& c, char ch)
{
    const auto u = static_cast<unsigned char>(ch);
    return (u < kVocabCap) ? c.char_to_class[u] : -1;
}

/// Class index -> character.  Returns '?' on out-of-range.
inline char ClassToChar(const Corpus& c, int cls)
{
    if (cls < 0 || static_cast<std::size_t>(cls) >= c.vocab.size()) return '?';
    return c.vocab[static_cast<std::size_t>(cls)];
}

/// Write 8 bipolar bits (+1 / -1) for the ASCII byte of c into out[0..7].
/// LSB first: out[0] = bit 0, out[7] = bit 7.
void BipolarBits(char c, float out[kInputBits]);

/// Bulk-encode a string into row-major bipolar bits: out has size s.size()*8.
void BipolarEncode(const std::string& s, float* out);

}  // namespace hrccnn_lm_text
