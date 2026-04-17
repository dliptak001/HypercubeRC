#pragma once

#include <cstdint>
#include <random>
#include <string>

namespace hrccnn_llm_math {

struct GeneratorConfig
{
    // Whether to apply the v1 training-set RHS filter |RHS| <= 999.
    bool rhs_filter_999 = true;

    // Probability of a leading unary minus on any sampled operand magnitude.
    double unary_minus_probability = 0.5;

    // 50/50 depth-1 vs depth-2. Within depth-2, the three structural forms
    // (left-paren, right-paren, both-paren) are sampled uniformly.
    double depth2_probability = 0.5;

    // Maximum operand magnitude for LHS atoms.
    double operand_max_magnitude = 100.0;
};

/// Random expression generator conforming to the HRCCNN_LLM_Math grammar.
///
/// A single `Sample()` call returns one canonical line including the trailing
/// '#'. Internally uses rejection sampling for divide-by-zero and, when
/// `rhs_filter_999` is enabled, for |RHS| > 999.
class Generator
{
public:
    explicit Generator(std::uint64_t seed, GeneratorConfig cfg = {});

    /// Generate one expression line, e.g. "(1.5 + 2.5) * 3 = 12#".
    /// May reject internally and retry; returns once a valid line is produced.
    std::string Sample();

private:
    std::mt19937_64  rng_;
    GeneratorConfig  cfg_;

    // Sample an atomic operand: magnitude in [0, operand_max_magnitude] at 2dp,
    // independent 50% unary-minus flip. Returns the signed double value and
    // its canonical printable form.
    struct Atom { double value; std::string text; };
    Atom SampleAtom();

    // Sample a binary operator uniformly from { +, -, *, / }.
    char SampleOp();

    // Try to generate a single depth-1 or depth-2 line. Returns empty string
    // on rejection (/0 or |RHS| > 999); caller retries.
    std::string TryOne();
};

}  // namespace hrccnn_llm_math
