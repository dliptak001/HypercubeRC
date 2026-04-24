#pragma once

/// @file Generator.h
/// @brief Autoregressive text generation with temperature-controlled sampling.
///
/// The Generator primes the reservoir on a prompt string, then produces
/// text one character at a time.  At each step it reads the CNN readout's
/// logits, samples the next character (greedy argmax or softmax sampling),
/// and feeds the chosen character back into the reservoir as input.

#include <cstddef>
#include <string>

#include "Model.h"
#include "Vocabulary.h"

namespace lm {

template <std::size_t DIM>
class Generator
{
public:
    /// @param model  The trained model to generate from.  The Generator
    ///               will reset and re-prime the reservoir on each call
    ///               to Generate().
    /// @param vocab  Vocabulary for decoding class indices to characters.
    Generator(Model<DIM>& model, const Vocabulary& vocab);

    /// Generate @p num_chars characters of text, starting from @p prompt.
    /// Returns only the generated suffix (not the prompt itself).
    ///
    /// @param temperature  Controls sampling randomness:
    ///   - 0.0 = greedy argmax (deterministic)
    ///   - 0.5–1.0 = typical range for readable text
    ///   - >1.0 = increasingly random
    /// @param seed  RNG seed for sampling (ignored when temperature == 0).
    [[nodiscard]] std::string Generate(const std::string& prompt,
                                       std::size_t num_chars,
                                       float temperature = 0.8f,
                                       unsigned seed = 42);

private:
    Model<DIM>&       model_;
    const Vocabulary&  vocab_;
};

}  // namespace lm
