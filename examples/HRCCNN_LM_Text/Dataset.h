#pragma once

#include <cstddef>
#include <string>

#include "Corpus.h"
#include "ESN.h"

namespace hrccnn_lm_text {

/// Reset the reservoir and stream `prompt` through it without collecting.
template <std::size_t DIM>
void ResetAndPrime(ESN<DIM>& esn, const std::string& prompt);

/// Autoregressively generate `num_chars` characters after priming on
/// `prompt`.  Returns only the generated suffix (not the prompt).
/// `temperature` controls sampling randomness:
///   0 = argmax (greedy), >0 = softmax sampling (0.5-1.0 typical).
template <std::size_t DIM>
std::string GenerateText(ESN<DIM>& esn,
                         const Corpus& corpus,
                         const std::string& prompt,
                         std::size_t num_chars,
                         float temperature = 0.0f,
                         unsigned seed = 42);

}  // namespace hrccnn_lm_text
