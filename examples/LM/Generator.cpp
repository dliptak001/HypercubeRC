#include "Generator.h"
#include "Config.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace lm {

template <std::size_t DIM>
Generator<DIM>::Generator(Model<DIM>& model, const Vocabulary& vocab)
    : model_(model), vocab_(vocab)
{
}

template <std::size_t DIM>
std::string Generator<DIM>::Generate(const std::string& prompt,
                                     std::size_t num_chars,
                                     float temperature,
                                     unsigned seed)
{
    // Prime: reset the reservoir and stream the prompt through it.
    model_.Reset();
    model_.Warmup(prompt);

    const std::size_t num_outputs = model_.NumOutputs();
    std::vector<float> logits(num_outputs);
    std::vector<float> probs(num_outputs);
    std::mt19937 rng(seed);

    std::string output;
    output.reserve(num_chars);

    for (std::size_t i = 0; i < num_chars; ++i) {
        model_.Predict(logits.data());

        std::size_t chosen;
        if (temperature <= 0.0f) {
            // Greedy argmax.
            chosen = 0;
            float best = logits[0];
            for (std::size_t k = 1; k < num_outputs; ++k) {
                if (logits[k] > best) {
                    best = logits[k];
                    chosen = k;
                }
            }
        } else {
            // Temperature-scaled softmax sampling.
            float max_logit = *std::max_element(logits.begin(), logits.end());
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

        char ch = vocab_.ClassToChar(static_cast<int>(chosen));
        output.push_back(ch);

        // Feed the generated character back into the reservoir.
        model_.Step(ch);
    }

    return output;
}

// =========================================================================
//  Explicit instantiation
// =========================================================================

using namespace lm::config;
template class Generator<kDIM>;

}  // namespace lm
