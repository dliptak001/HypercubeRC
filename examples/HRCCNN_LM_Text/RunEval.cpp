#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "Config.h"
#include "Corpus.h"
#include "Dataset.h"
#include "ESN.h"
#include "Serialization.h"

namespace hrccnn_lm_text {

using config::kDIM;

int RunEval()
{
    const config::EvalCfg& args = config::kEval;

    if (args.model_path.empty()) {
        std::cerr << "error: config::kEval.model_path is empty\n"; return 1;
    }
    if (args.corpus_path.empty()) {
        std::cerr << "error: config::kEval.corpus_path is empty\n"; return 1;
    }

    std::string err;
    ModelFile mf;
    if (!LoadModelFile(args.model_path, mf, &err)) {
        std::cerr << "error: failed to load model: " << err << "\n"; return 4;
    }
    if (mf.dim != kDIM) {
        std::cerr << "error: model DIM=" << mf.dim
                  << " does not match binary DIM=" << kDIM << "\n";
        return 4;
    }

    // Load corpus + pin vocab to model's saved vocab.
    Corpus corpus;
    if (!LoadCorpus(args.corpus_path, corpus)) {
        std::cerr << "error: could not load corpus from " << args.corpus_path << "\n";
        return 2;
    }
    AttachCorpus(corpus.text, mf.vocab, corpus);

    const std::size_t total_needed =
        args.warmup_chars + args.skip_chars + args.eval_chars + 1;
    if (corpus.text.size() < total_needed) {
        std::cerr << "error: corpus has " << corpus.text.size()
                  << " chars, need " << total_needed
                  << " (warmup+skip+eval+1)\n";
        return 2;
    }

    std::cerr << "[eval] model=" << args.model_path
              << " (trained positions=" << mf.meta.training_positions
              << " epochs=" << mf.meta.training_epochs
              << " seed=" << mf.meta.training_seed << ")\n";
    std::cerr << "[eval] warmup=" << args.warmup_chars
              << " skip=" << args.skip_chars
              << " eval=" << args.eval_chars << "\n";

    // --- Construct ESN + restore weights (standard bootstrap dance). ---
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

    // --- Encode + drive. ---
    //     Warmup covers warmup_chars + skip_chars (no state collected).
    //     Run collects only eval_chars states — the ones we actually score.
    std::vector<float> bits(total_needed * kInputBits);
    for (std::size_t t = 0; t < total_needed; ++t) {
        const char c = corpus.text[t];
        if (CharToClass(corpus, c) < 0) {
            std::cerr << "error: eval corpus char at offset " << t
                      << " (byte " << static_cast<int>(static_cast<unsigned char>(c))
                      << ") not in model vocab\n";
            return 2;
        }
        BipolarBits(c, bits.data() + t * kInputBits);
    }

    const std::size_t warmup_total = args.warmup_chars + args.skip_chars;

    esn.ResetReservoirOnly();
    if (warmup_total > 0)
        esn.Warmup(bits.data(), warmup_total);

    esn.Run(bits.data() + warmup_total * kInputBits, args.eval_chars);

    // Targets: one per collected state (eval_chars only).
    const std::size_t eval_start = warmup_total;
    std::vector<float> targets(args.eval_chars);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        targets[i] = static_cast<float>(
            CharToClass(corpus, corpus.text[eval_start + i + 1]));
    }

    const double char_acc = esn.Accuracy(targets.data(), 0, args.eval_chars);
    std::cerr << "[eval] char_accuracy (teacher-forced) = " << char_acc << "\n";
    return 0;
}

}  // namespace hrccnn_lm_text
