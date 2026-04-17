#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "Dataset.h"
#include "ESN.h"
#include "Serialization.h"
#include "Vocab.h"

namespace hrccnn_llm_math {

namespace {

constexpr std::size_t kDIM = 12;
constexpr std::size_t kMaxOutputChars = 16;

struct InferArgs
{
    std::string model_path;
    std::string input;
    std::size_t max_output_chars = kMaxOutputChars;
};

bool ParseArgs(int argc, char** argv, InferArgs& a)
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
        if      (f == "--model")      { auto v = next(f.c_str()); if (!v) return false; a.model_path = v; }
        else if (f == "--input")      { auto v = next(f.c_str()); if (!v) return false; a.input = v; }
        else if (f == "--max-output") { auto v = next(f.c_str()); if (!v) return false; a.max_output_chars = std::strtoull(v, nullptr, 10); }
        else {
            std::cerr << "error: unknown flag '" << f << "'\n";
            return false;
        }
    }
    if (a.model_path.empty()) {
        std::cerr << "error: --model <path> is required\n";
        return false;
    }
    if (a.input.empty()) {
        std::cerr << "error: --input \"<LHS>\" is required\n";
        return false;
    }
    // Reject any character not in the vocab — catches typos early.
    for (char c : a.input) {
        if (CharToClass(c) < 0) {
            std::cerr << "error: input character '" << c
                      << "' is not in the vocab\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int RunInfer(int argc, char** argv)
{
    InferArgs args;
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

    ESN<kDIM> esn(mf.reservoir_cfg, ReadoutType::HCNN, FeatureMode::Raw);
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

    std::string emitted = GenerateRHS(esn, args.input, args.max_output_chars);
    std::cout << args.input << " = " << emitted;
    if (emitted.empty() || emitted.back() != '#') std::cout << "  [no-stop]";
    std::cout << "\n";
    return 0;
}

}  // namespace hrccnn_llm_math
