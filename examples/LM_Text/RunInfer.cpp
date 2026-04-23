#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "Config.h"
#include "Corpus.h"
#include "Dataset.h"
#include "ESN.h"
#include "Serialization.h"

namespace lm_text {

using config::kDIM;

int RunInfer()
{
    const config::InferCfg& args = config::kInfer;

    if (args.model_path.empty()) {
        std::cerr << "error: config::kInfer.model_path is empty\n";
        return 1;
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

    // No corpus needed at infer time — vocab comes from the model file.
    Corpus corpus;
    AttachCorpus(/*text=*/{}, mf.vocab, corpus);

    // Reject prompt chars that aren't in the model's vocab.
    for (char c : args.prompt) {
        if (CharToClass(corpus, c) < 0) {
            std::cerr << "error: prompt contains char outside model vocab: '"
                      << c << "' (byte " << static_cast<int>(
                         static_cast<unsigned char>(c)) << ")\n";
            return 3;
        }
    }

    // --- ESN construction + weight restore. ---
    ESN<kDIM> esn(mf.reservoir_cfg);
    esn.SetCNNConfig(mf.cnn_cfg);
    esn.SetReadoutState(FromSerial<kDIM>(mf.readout));

    std::cerr << "[infer] model=" << args.model_path
              << " vocab_size=" << mf.vocab.size()
              << " prompt_len=" << args.prompt.size()
              << " gen_chars=" << args.num_chars << "\n";

    const std::string gen = GenerateText(esn, corpus, args.prompt, args.num_chars,
                                           args.temperature, args.gen_seed);
    std::cout << args.prompt << gen;
    if (gen.empty() || gen.back() != '\n') std::cout << '\n';
    return 0;
}

}  // namespace lm_text
