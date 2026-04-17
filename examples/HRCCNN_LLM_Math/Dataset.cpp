#include "Dataset.h"

#include <cstddef>
#include <string>
#include <vector>

#include "Vocab.h"

namespace hrccnn_llm_math {

bool SplitLine(const std::string& line, LineSplit& out)
{
    if (line.empty() || line.back() != '#') return false;
    auto eq = line.find(" = ");
    if (eq == std::string::npos) return false;
    out.full = line;
    out.lhs  = line.substr(0, eq);
    out.rhs  = line.substr(eq + 3, line.size() - eq - 3 - 1);  // drop trailing '#'
    return true;
}

template <size_t DIM>
void ResetAndPrime(ESN<DIM>& esn, const std::string& lhs)
{
    esn.ResetReservoirOnly();
    if (lhs.empty()) return;
    std::vector<float> bits(lhs.size() * kInputBits);
    BipolarEncode(lhs, bits.data());
    // Two priming passes over the LHS, no state collection.
    esn.Warmup(bits.data(), lhs.size());
    esn.Warmup(bits.data(), lhs.size());
}

template <size_t DIM>
void TeacherForceOne(ESN<DIM>& esn,
                     const LineSplit& split,
                     std::vector<float>& targets_out)
{
    const std::string& full = split.full;
    const std::size_t  L    = full.size();
    if (L < 2) return;  // nothing to train on

    // Validate every character is in the vocab BEFORE touching the reservoir.
    // Silently skipping bad lines here keeps train targets aligned with
    // collected snapshots downstream (asserts are stripped in Release, so
    // any escaped bad char would otherwise produce target=-1 garbage).
    for (std::size_t t = 1; t < L; ++t) {
        if (CharToClass(full[t]) < 0) return;
    }

    ResetAndPrime(esn, split.lhs);

    // Run for (L - 1) steps so snapshot[t] = state-after-char(t), paired with
    // target char(t + 1). We feed char(0)..char(L-2) into the reservoir; the
    // final char (always '#') never needs to be "consumed" as an input.
    std::vector<float> bits((L - 1) * kInputBits);
    for (std::size_t t = 0; t + 1 < L; ++t) {
        BipolarBits(full[t], bits.data() + t * kInputBits);
    }
    esn.Run(bits.data(), L - 1);

    targets_out.reserve(targets_out.size() + (L - 1));
    for (std::size_t t = 1; t < L; ++t) {
        targets_out.push_back(static_cast<float>(CharToClass(full[t])));
    }
}

template <size_t DIM>
std::string GenerateRHS(ESN<DIM>& esn,
                        const std::string& lhs,
                        std::size_t max_output_chars)
{
    ResetAndPrime(esn, lhs);

    // Final LHS pass (the generation protocol primes 2x and then does one
    // collecting-less final pass so the reservoir is at "just consumed the
    // last LHS char" when the argmax loop begins).
    if (!lhs.empty()) {
        std::vector<float> bits(lhs.size() * kInputBits);
        BipolarEncode(lhs, bits.data());
        esn.Warmup(bits.data(), lhs.size());
    }

    const std::size_t num_outputs = esn.NumOutputs();
    std::vector<float> logits(num_outputs);
    std::string out;
    out.reserve(max_output_chars);

    for (std::size_t i = 0; i < max_output_chars; ++i) {
        esn.PredictLiveRaw(logits.data());
        // argmax over raw logits (softmax doesn't change argmax).
        std::size_t best = 0;
        float best_v = logits[0];
        for (std::size_t k = 1; k < num_outputs; ++k) {
            if (logits[k] > best_v) { best_v = logits[k]; best = k; }
        }
        char c = ClassToChar(static_cast<int>(best));
        out.push_back(c);
        if (c == '#') break;

        float step_bits[kInputBits];
        BipolarBits(c, step_bits);
        esn.Warmup(step_bits, 1);
    }
    return out;
}

// Explicit instantiations for the only DIM this example currently targets.
// If you add a new DIM, add it here too.
template void ResetAndPrime<12>(ESN<12>&, const std::string&);
template void TeacherForceOne<12>(ESN<12>&, const LineSplit&, std::vector<float>&);
template std::string GenerateRHS<12>(ESN<12>&, const std::string&, std::size_t);

}  // namespace hrccnn_llm_math
