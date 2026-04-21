#include <iostream>
#include <string>
#include <vector>

#include "Config.h"
#include "Dataset.h"
#include "ESN.h"
#include "Serialization.h"
#include "Vocab.h"

namespace hrccnn_llm_math {

namespace {

constexpr std::size_t kDIM = 12;

}  // namespace

int RunInfer()
{
    const config::InferCfg& args = config::kInfer;

    if (args.model_path.empty()) {
        std::cerr << "error: config::kInfer.model_path is empty\n";
        return 1;
    }
    if (args.input.empty()) {
        std::cerr << "error: config::kInfer.input is empty\n";
        return 1;
    }
    for (char c : args.input) {
        if (CharToClass(c) < 0) {
            std::cerr << "error: input character '" << c
                      << "' is not in the vocab\n";
            return 1;
        }
    }

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

    ESN<kDIM> esn(mf.reservoir_cfg, ReadoutType::HCNN);
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
