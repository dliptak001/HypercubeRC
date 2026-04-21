#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
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
              << " passes=" << mf.meta.training_passes
              << " seed=" << mf.meta.training_seed << ")\n";
    std::cerr << "[eval] streaming: warmup=" << args.warmup_chars
              << " skip=" << args.skip_chars
              << " eval=" << args.eval_chars << "\n";

    // --- Construct ESN + restore weights. ---
    ESN<kDIM> esn(mf.reservoir_cfg);
    esn.SetCNNConfig(mf.cnn_cfg);
    esn.SetReadoutState(FromSerial<kDIM>(mf.readout));

    // --- Phase 1: Stream warmup + skip region (no scoring). ---
    esn.ResetReservoirOnly();
    float step_bits[kInputBits];
    std::size_t corpus_pos = 0;

    const std::size_t warmup_total = args.warmup_chars + args.skip_chars;
    for (std::size_t i = 0; i < warmup_total; ++i) {
        BipolarBits(corpus.text[corpus_pos++], step_bits);
        esn.Warmup(step_bits, 1);
    }

    // --- Phase 2: Stream eval region, scoring each step. ---
    const std::size_t num_outputs = esn.NumOutputs();
    std::vector<float> logits(num_outputs);
    std::vector<std::size_t> sorted_idx(num_outputs);

    std::size_t correct1 = 0, correct3 = 0, correct5 = 0;
    double total_log_loss = 0.0;
    const std::size_t num_classes = kVocabSize;
    std::vector<std::size_t> per_class_correct(num_classes, 0);
    std::vector<std::size_t> per_class_total(num_classes, 0);

    for (std::size_t i = 0; i < args.eval_chars; ++i) {
        BipolarBits(corpus.text[corpus_pos], step_bits);
        esn.Warmup(step_bits, 1);

        esn.PredictLiveRaw(logits.data());
        const int label = CharToClass(corpus, corpus.text[corpus_pos + 1]);

        float max_logit = *std::max_element(logits.begin(),
                                            logits.begin() + static_cast<long>(num_outputs));
        double sum_exp = 0.0;
        for (std::size_t k = 0; k < num_outputs; ++k)
            sum_exp += std::exp(static_cast<double>(logits[k]) - max_logit);
        double log_prob = (logits[label] - max_logit) - std::log(sum_exp);
        total_log_loss -= log_prob;

        std::iota(sorted_idx.begin(), sorted_idx.end(), std::size_t{0});
        std::size_t k_max = std::min<std::size_t>(5, num_outputs);
        std::partial_sort(sorted_idx.begin(),
                          sorted_idx.begin() + static_cast<long>(k_max),
                          sorted_idx.end(),
                          [&](std::size_t a, std::size_t b) {
                              return logits[a] > logits[b];
                          });

        for (std::size_t k = 0; k < k_max; ++k) {
            if (static_cast<int>(sorted_idx[k]) == label) {
                if (k < 1) ++correct1;
                if (k < 3) ++correct3;
                if (k < 5) ++correct5;
                break;
            }
        }

        if (label >= 0 && static_cast<std::size_t>(label) < num_classes) {
            per_class_total[label]++;
            if (static_cast<int>(sorted_idx[0]) == label)
                per_class_correct[label]++;
        }

        ++corpus_pos;
    }

    const double top1 = static_cast<double>(correct1) / args.eval_chars;
    const double top3 = static_cast<double>(correct3) / args.eval_chars;
    const double top5 = static_cast<double>(correct5) / args.eval_chars;
    const double bpc  = total_log_loss / (args.eval_chars * std::log(2.0));

    std::cerr << "[eval] top1=" << top1
              << " top3=" << top3
              << " top5=" << top5
              << " bpc=" << bpc << "\n";

    // Per-class worst-N breakdown.
    if (args.eval_worst_classes > 0) {
        std::vector<std::size_t> idx(num_classes);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            double acc_a = per_class_total[a] > 0
                ? static_cast<double>(per_class_correct[a]) / per_class_total[a] : 2.0;
            double acc_b = per_class_total[b] > 0
                ? static_cast<double>(per_class_correct[b]) / per_class_total[b] : 2.0;
            return acc_a < acc_b;
        });

        std::cerr << "[eval] worst " << args.eval_worst_classes << " classes:";
        std::size_t printed = 0;
        for (std::size_t i = 0; i < idx.size() && printed < args.eval_worst_classes; ++i) {
            std::size_t ci = idx[i];
            if (per_class_total[ci] == 0) continue;
            double acc = static_cast<double>(per_class_correct[ci]) / per_class_total[ci];
            char ch = ClassToChar(corpus, static_cast<int>(ci));
            std::string repr;
            if (ch == '\n') repr = "\\n";
            else if (ch == '\r') repr = "\\r";
            else if (ch == '\t') repr = "\\t";
            else if (ch == ' ') repr = "SP";
            else { repr = "'"; repr += ch; repr += "'"; }
            std::cerr << " " << repr << "=" << static_cast<int>(acc * 100) << "%"
                      << "(" << per_class_correct[ci] << "/" << per_class_total[ci] << ")";
            ++printed;
        }
        std::cerr << "\n";
    }

    return 0;
}

}  // namespace hrccnn_lm_text
