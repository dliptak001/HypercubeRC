#pragma once

/// @file Corpus.h
/// @brief Text corpus validated against a Vocabulary.
///
/// A Corpus is an immutable block of text that has been validated to contain
/// only characters present in the Vocabulary.  It provides indexed access to
/// the raw text for streaming through the reservoir.

#include <cstddef>
#include <string>

#include "Vocabulary.h"

namespace lm {

class Corpus
{
public:
    /// Load a corpus from a text file.  Every byte in the file must be
    /// present in @p vocab; throws std::runtime_error on missing file,
    /// empty file, or any byte outside the vocabulary.
    static Corpus LoadFile(const std::string& path, const Vocabulary& vocab);

    /// Construct a corpus from an existing string (no file I/O).
    /// Does not validate — caller is responsible for ensuring all
    /// characters are in the vocabulary.
    explicit Corpus(std::string text);

    [[nodiscard]] const std::string& Text()          const { return text_; }
    [[nodiscard]] std::size_t        Size()          const { return text_.size(); }
    [[nodiscard]] char               At(std::size_t pos) const { return text_[pos]; }

private:
    std::string text_;
};

}  // namespace lm
