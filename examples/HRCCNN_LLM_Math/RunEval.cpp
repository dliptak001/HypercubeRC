#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include "Dataset.h"
#include "ESN.h"
#include "Generator.h"
#include "Serialization.h"
#include "Vocab.h"

namespace hrccnn_llm_math {

namespace {

constexpr std::size_t kDIM = 12;
constexpr std::size_t kMaxOutputChars = 16;

struct EvalArgs
{
    std::string      model_path;
    std::size_t      samples   = 2000;
    std::uint64_t    seed      = 0;
    bool             seed_set  = false;
    bool             skip_char_accuracy = false;  ///< Skip the teacher-forced pass (saves memory/time).
};

bool ParseArgs(int argc, char** argv, EvalArgs& a)
{
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "error: " << name << " requires a value\n";
                return nullptr;
            }
            return argv[++i];
        };
        if      (f == "--model")   { auto v = next(f.c_str()); if (!v) return false; a.model_path = v; }
        else if (f == "--samples") { auto v = next(f.c_str()); if (!v) return false; a.samples = std::strtoull(v, nullptr, 10); }
        else if (f == "--seed")    { auto v = next(f.c_str()); if (!v) return false; a.seed = std::strtoull(v, nullptr, 10); a.seed_set = true; }
        else if (f == "--no-char") { a.skip_char_accuracy = true; }
        else {
            std::cerr << "error: unknown flag '" << f << "'\n";
            return false;
        }
    }
    if (a.model_path.empty()) {
        std::cerr << "error: --model <path> is required\n";
        return false;
    }
    return true;
}

bool IsValidFormat(const std::string& emitted)
{
    static const std::regex re(R"(^-?(0|[1-9]\d{0,8})(\.\d|\.\d[1-9])?#$)");
    return std::regex_match(emitted, re);
}

}  // namespace

int RunEval(int argc, char** argv)
{
    EvalArgs args;
    if (!ParseArgs(argc, argv, args)) return 1;

    std::string err;
    ModelFile mf;
    if (!LoadModelFile(args.model_path, mf, &err)) {
        std::cerr << "error: failed to load model: " << err << "\n";
        return 4;
    }
    if (mf.dim != kDIM) {
        std::cerr << "error: model DIM=" << mf.dim
                  << " does not match binary DIM=" << kDIM << "\n";
        return 4;
    }

    if (!args.seed_set) {
        std::random_device rd;
        args.seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }

    std::cerr << "[eval] model=" << args.model_path
              << " (trained samples=" << mf.meta.training_samples
              << " epochs=" << mf.meta.training_epochs
              << " seed=" << mf.meta.training_seed;
    if (!mf.meta.git_sha.empty()) std::cerr << " git=" << mf.meta.git_sha;
    std::cerr << ")\n";
    std::cerr << "[eval] eval_samples=" << args.samples
              << " eval_seed=" << args.seed << "\n";

    ESN<kDIM> esn(mf.reservoir_cfg, ReadoutType::HCNN, FeatureMode::Raw);
    // Bootstrap the CNN topology. CNNReadout builds its architecture lazily
    // inside Train(), so a no-op train with epochs=0 on a single dummy state
    // establishes the HCNN net and matching scratch buffers. SetReadoutState
    // then overwrites the randomly-initialized weights and the garbage
    // (single-sample) standardization vectors with the saved values.
    {
        std::vector<float> dummy_bits(kInputBits, 0.0f);
        esn.Run(dummy_bits.data(), 1);
        float dummy_target = 0.0f;
        CNNReadoutConfig bootstrap = mf.cnn_cfg;
        bootstrap.epochs  = 0;
        bootstrap.verbose = false;
        esn.Train(&dummy_target, 1, bootstrap);
        esn.ClearStates();
    }
    esn.SetReadoutState(FromSerial<kDIM>(mf.readout));

    GeneratorConfig gcfg;
    gcfg.rhs_filter_999 = true;

    // --- Generate eval lines. ---
    Generator gen(args.seed, gcfg);
    std::vector<std::string> eval_lines;
    eval_lines.reserve(args.samples);
    for (std::size_t i = 0; i < args.samples; ++i) eval_lines.push_back(gen.Sample());

    // --- Phase A: teacher-forced char accuracy. ---
    double char_acc = 0.0;
    if (!args.skip_char_accuracy) {
        std::vector<float> targets;
        for (const auto& line : eval_lines) {
            LineSplit sp;
            if (!SplitLine(line, sp)) continue;
            TeacherForceOne(esn, sp, targets);
        }
        if (esn.NumCollected() > 0) {
            char_acc = esn.Accuracy(targets.data(), 0, esn.NumCollected());
        }
        esn.ClearStates();
    }

    // --- Phase B: autoregressive scoring. ---
    std::size_t exact = 0, format_ok = 0, non_stop = 0;
    for (const auto& line : eval_lines) {
        LineSplit sp;
        if (!SplitLine(line, sp)) continue;
        std::string emitted = GenerateRHS(esn, sp.lhs, kMaxOutputChars);
        bool stopped_on_hash = !emitted.empty() && emitted.back() == '#';
        if (!stopped_on_hash) ++non_stop;
        if (IsValidFormat(emitted)) ++format_ok;
        std::string emitted_rhs =
            stopped_on_hash ? emitted.substr(0, emitted.size() - 1) : emitted;
        if (emitted_rhs == sp.rhs && stopped_on_hash) ++exact;
    }

    const std::size_t n = eval_lines.size();
    std::cerr << "[eval] results over " << n << " expressions:\n";
    if (!args.skip_char_accuracy) {
        std::cerr << "  char_accuracy  (teacher-forced) = " << char_acc << "\n";
    }
    std::cerr << "  format_accuracy                  = "
              << (static_cast<double>(format_ok) / static_cast<double>(n))
              << "   (" << format_ok << "/" << n << ")\n";
    std::cerr << "  exact_match_accuracy             = "
              << (static_cast<double>(exact) / static_cast<double>(n))
              << "   (" << exact << "/" << n << ")\n";
    std::cerr << "  non_stop (no '#' emitted)        = "
              << non_stop << "/" << n << "\n";
    return 0;
}

}  // namespace hrccnn_llm_math
