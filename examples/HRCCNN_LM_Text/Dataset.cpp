#include "Dataset.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace hrccnn_lm_text {

template <std::size_t DIM>
void ResetAndPrime(ESN<DIM>& esn, const std::string& prompt)
{
    esn.ResetReservoirOnly();
    if (prompt.empty()) return;
    std::vector<float> bits(prompt.size() * kInputBits);
    BipolarEncode(prompt, bits.data());
    esn.Warmup(bits.data(), prompt.size());
}

template <std::size_t DIM>
std::string GenerateText(ESN<DIM>& esn,
                         const Corpus& corpus,
                         const std::string& prompt,
                         std::size_t num_chars,
                         float temperature,
                         unsigned seed)
{
    ResetAndPrime(esn, prompt);

    const std::size_t num_outputs = esn.NumOutputs();
    std::vector<float> logits(num_outputs);
    std::string out;
    out.reserve(num_chars);

    std::mt19937 rng(seed);

    float step_bits[kInputBits];
    for (std::size_t i = 0; i < num_chars; ++i) {
        esn.PredictLiveRaw(logits.data());

        std::size_t chosen;
        if (temperature <= 0.0f) {
            chosen = 0;
            float best_v = logits[0];
            for (std::size_t k = 1; k < num_outputs; ++k) {
                if (logits[k] > best_v) { best_v = logits[k]; chosen = k; }
            }
        } else {
            float max_logit = *std::max_element(logits.begin(), logits.end());
            std::vector<float> probs(num_outputs);
            float sum = 0.0f;
            for (std::size_t k = 0; k < num_outputs; ++k) {
                probs[k] = std::exp((logits[k] - max_logit) / temperature);
                sum += probs[k];
            }
            for (std::size_t k = 0; k < num_outputs; ++k)
                probs[k] /= sum;

            std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
            chosen = dist(rng);
        }

        const char ch = ClassToChar(corpus, static_cast<int>(chosen));
        out.push_back(ch);

        BipolarBits(ch, step_bits);
        esn.Warmup(step_bits, 1);
    }
    return out;
}

// DIM 12 only; add more if targeting other sizes.
template void ResetAndPrime<12>(ESN<12>&, const std::string&);
template std::string GenerateText<12>(ESN<12>&, const Corpus&,
                                      const std::string&, std::size_t,
                                      float, unsigned);

}  // namespace hrccnn_lm_text
