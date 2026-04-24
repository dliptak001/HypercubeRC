/// @file main.cpp
/// @brief Entry point for the LM character-level language model.
///
/// Dispatches to Train, Eval, or Infer based on config::kMode.
/// Each mode constructs the appropriate objects and delegates to them.

#include <iostream>
#include <random>
#include <string>

#include "Config.h"
#include "Corpus.h"
#include "Evaluator.h"
#include "Generator.h"
#include "Model.h"
#include "Trainer.h"
#include "Vocabulary.h"
#include "Presets.h"

namespace lm {

using config::kDIM;

// -------------------------------------------------------------------------
//  Train mode
// -------------------------------------------------------------------------

static int RunTrain()
{
    const auto& args = config::kTrain;

    if (args.corpus_path.empty() || args.output_path.empty()) {
        std::cerr << "error: corpus_path and output_path must be set\n";
        return 1;
    }

    Vocabulary vocab;
    Corpus corpus = Corpus::LoadFile(args.corpus_path, vocab);

    // Derive reservoir seed from generation seed.
    std::uint64_t gen_seed = args.gen_seed;
    if (!args.use_fixed_gen_seed) {
        std::random_device rd;
        gen_seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }
    const std::uint64_t reservoir_seed = args.use_fixed_reservoir_seed
        ? args.reservoir_seed
        : (gen_seed ^ 0x9E3779B97F4A7C15ULL);

    ReservoirConfig rcfg;
    rcfg.seed            = reservoir_seed;
    rcfg.num_inputs      = Vocabulary::kInputBits;
    rcfg.spectral_radius = args.spectral_radius;
    rcfg.leak_rate       = args.leak_rate;
    rcfg.output_fraction = args.output_fraction;

    ReadoutArchConfig arch = presets::Baseline<kDIM>().arch;
    arch.task          = ReadoutTask::Classification;
    arch.num_outputs   = static_cast<int>(Vocabulary::kSize);
    arch.num_layers    = args.cnn_num_layers;
    arch.conv_channels = args.cnn_conv_channels;

    Model<kDIM> model(rcfg, arch);

    Trainer<kDIM> trainer(model, corpus, vocab, args);
    return trainer.Run();
}

// -------------------------------------------------------------------------
//  Eval mode
// -------------------------------------------------------------------------

static int RunEval()
{
    const auto& args = config::kEval;

    Vocabulary vocab;
    Model<kDIM> model = Model<kDIM>::Load(args.model_path, vocab);
    Corpus corpus = Corpus::LoadFile(args.corpus_path, vocab);

    const std::size_t total_needed =
        args.warmup_chars + args.skip_chars + args.eval_chars + 1;
    if (corpus.Size() < total_needed) {
        std::cerr << "error: corpus has " << corpus.Size()
                  << " chars, need " << total_needed << "\n";
        return 2;
    }

    std::cerr << "[eval] model=" << args.model_path
              << "  warmup=" << args.warmup_chars
              << "  skip=" << args.skip_chars
              << "  eval=" << args.eval_chars << "\n";

    // Stream warmup + skip (no scoring).
    model.Reset();
    std::size_t pos = 0;
    for (std::size_t i = 0; i < args.warmup_chars + args.skip_chars; ++i)
        model.Step(corpus.At(pos++));

    // Stream eval region, scoring each step.
    Evaluator eval(Vocabulary::kSize);
    std::vector<float> logits(model.NumOutputs());

    for (std::size_t i = 0; i < args.eval_chars; ++i) {
        model.Step(corpus.At(pos));
        model.Predict(logits.data());

        int label = vocab.CharToClass(corpus.At(pos + 1));
        eval.Record(logits.data(), label);
        ++pos;
    }

    Metrics m = eval.Compute();
    std::cerr << Evaluator::FormatMetrics("eval", m) << "\n";

    auto worst = eval.WorstClasses(args.eval_worst_classes);
    if (!worst.empty())
        std::cerr << Evaluator::FormatWorstClasses("eval", worst, vocab) << "\n";

    return 0;
}

// -------------------------------------------------------------------------
//  Infer mode
// -------------------------------------------------------------------------

static int RunInfer()
{
    const auto& args = config::kInfer;

    Vocabulary vocab;
    Model<kDIM> model = Model<kDIM>::Load(args.model_path, vocab);

    // Validate that every prompt character is in the vocabulary.
    for (char c : args.prompt) {
        if (!vocab.Contains(c)) {
            std::cerr << "error: prompt contains char outside vocab: '"
                      << c << "' (0x" << std::hex
                      << static_cast<int>(static_cast<unsigned char>(c))
                      << std::dec << ")\n";
            return 3;
        }
    }

    std::cerr << "[infer] model=" << args.model_path
              << "  prompt_len=" << args.prompt.size()
              << "  gen_chars=" << args.num_chars << "\n";

    Generator<kDIM> gen(model, vocab);
    std::string text = gen.Generate(args.prompt, args.num_chars,
                                    args.temperature, args.gen_seed);

    std::cout << args.prompt << text;
    if (text.empty() || text.back() != '\n')
        std::cout << '\n';

    return 0;
}

}  // namespace lm

// =========================================================================

int main()
{
    using namespace lm;
    switch (config::kMode) {
        case config::Mode::Train: return RunTrain();
        case config::Mode::Eval:  return RunEval();
        case config::Mode::Infer: return RunInfer();
        default:
            std::cerr << "error: unknown config::kMode\n";
            return 1;
    }
}
