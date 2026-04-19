#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "Config.h"
#include "Corpus.h"
#include "Dataset.h"
#include "ESN.h"
#include "Serialization.h"
#include "readout/HCNNPresets.h"

namespace hrccnn_lm_text {

using config::kDIM;

namespace {

std::string EscapeText(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if      (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    return out;
}

struct EvalMetrics {
    double top1     = 0.0;
    double top3     = 0.0;
    double top5     = 0.0;
    double bpc      = 0.0;
    std::vector<std::size_t> per_class_correct;
    std::vector<std::size_t> per_class_total;
};

void PrintMetrics(const std::string& tag,
                  const std::string& split,
                  const EvalMetrics& m,
                  const Corpus& corpus,
                  std::size_t worst_n)
{
    std::cerr << "[" << tag << "] " << split
              << ": top1=" << m.top1
              << " top3=" << m.top3
              << " top5=" << m.top5
              << " bpc=" << m.bpc << "\n";

    if (worst_n == 0 || m.per_class_total.empty()) return;

    std::vector<std::size_t> idx(m.per_class_total.size());
    std::iota(idx.begin(), idx.end(), 0);
    // Sort by accuracy ascending (worst first), skip classes with 0 samples.
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        double acc_a = m.per_class_total[a] > 0
            ? static_cast<double>(m.per_class_correct[a]) / m.per_class_total[a] : 2.0;
        double acc_b = m.per_class_total[b] > 0
            ? static_cast<double>(m.per_class_correct[b]) / m.per_class_total[b] : 2.0;
        return acc_a < acc_b;
    });

    std::cerr << "[" << tag << "] worst " << worst_n << " classes:";
    std::size_t printed = 0;
    for (std::size_t i = 0; i < idx.size() && printed < worst_n; ++i) {
        std::size_t ci = idx[i];
        if (m.per_class_total[ci] == 0) continue;
        double acc = static_cast<double>(m.per_class_correct[ci]) / m.per_class_total[ci];
        char ch = ClassToChar(corpus, static_cast<int>(ci));
        std::string repr;
        if (ch == '\n') repr = "\\n";
        else if (ch == '\r') repr = "\\r";
        else if (ch == '\t') repr = "\\t";
        else if (ch == ' ') repr = "SP";
        else { repr = "'"; repr += ch; repr += "'"; }
        std::cerr << " " << repr << "=" << static_cast<int>(acc * 100) << "%"
                  << "(" << m.per_class_correct[ci] << "/" << m.per_class_total[ci] << ")";
        ++printed;
    }
    std::cerr << "\n";
}

}  // namespace

int RunTrain()
{
    const config::TrainCfg& args = config::kTrain;

    if (args.output_path.empty()) {
        std::cerr << "error: config::kTrain.output_path is empty\n"; return 1;
    }
    if (args.corpus_path.empty()) {
        std::cerr << "error: config::kTrain.corpus_path is empty\n"; return 1;
    }

    Corpus corpus;
    if (!LoadCorpus(args.corpus_path, corpus)) {
        std::cerr << "error: could not load corpus from " << args.corpus_path << "\n";
        return 2;
    }
    std::cerr << "[train] corpus=" << args.corpus_path
              << " chars=" << corpus.text.size()
              << " vocab_size=" << corpus.vocab.size() << "\n";

    const std::size_t total_chars =
        args.warmup_chars + args.warmup_train_chars +
        args.train_chars + args.val_chars + 1;
    if (corpus.text.size() < total_chars) {
        std::cerr << "error: corpus has " << corpus.text.size()
                  << " chars, need " << total_chars << "\n";
        return 2;
    }

    std::uint64_t gen_seed = args.gen_seed;
    if (!args.use_fixed_gen_seed) {
        std::random_device rd;
        gen_seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }
    const std::uint64_t reservoir_seed = args.use_fixed_reservoir_seed
                                             ? args.reservoir_seed
                                             : (gen_seed ^ 0x9E3779B97F4A7C15ULL);

    ReservoirConfig rcfg;
    rcfg.seed             = reservoir_seed;
    rcfg.num_inputs       = kInputBits;
    rcfg.spectral_radius  = args.spectral_radius;
    rcfg.output_fraction  = args.output_fraction;

    const std::size_t N = (1ULL << kDIM);
    std::cerr << "[train] DIM=" << kDIM << " N=" << N
              << " output_fraction=" << rcfg.output_fraction
              << " reservoir_seed=" << rcfg.seed << "\n";
    std::cerr << "[train] streaming: warmup=" << args.warmup_chars
              << " warmup_train=" << args.warmup_train_chars
              << " train=" << args.train_chars
              << " val=" << args.val_chars << "\n";

    ESN<kDIM> esn(rcfg, ReadoutType::HCNN, FeatureMode::Raw);

    // --- Phase 1: Reservoir warmup (no collection). ---
    std::size_t corpus_pos = 0;
    float step_bits[kInputBits];

    esn.ResetReservoirOnly();
    for (std::size_t i = 0; i < args.warmup_chars; ++i) {
        BipolarBits(corpus.text[corpus_pos++], step_bits);
        esn.Warmup(step_bits, 1);
    }

    // --- Phase 2: Collect warmup_train_chars for CNN standardization. ---
    std::vector<float> warmup_bits(args.warmup_train_chars * kInputBits);
    for (std::size_t i = 0; i < args.warmup_train_chars; ++i)
        BipolarBits(corpus.text[corpus_pos + i], warmup_bits.data() + i * kInputBits);

    CNNReadoutConfig cnn_cfg = hcnn_presets::HRCCNNBaseline<kDIM>();
    cnn_cfg.task          = HCNNTask::Classification;
    cnn_cfg.num_outputs   = static_cast<int>(kVocabSize);
    cnn_cfg.num_layers    = args.cnn_num_layers;
    cnn_cfg.conv_channels = args.cnn_conv_channels;

    esn.InitOnline(warmup_bits.data(), args.warmup_train_chars, cnn_cfg);
    warmup_bits.clear();
    warmup_bits.shrink_to_fit();

    // InitOnline's Run() already advanced the reservoir through the
    // warmup_train region.  Just advance corpus_pos to match.
    corpus_pos += args.warmup_train_chars;

    std::cerr << "[train] CNN cfg: nl=" << cnn_cfg.num_layers
              << " ch=" << cnn_cfg.conv_channels
              << " head=" << (cnn_cfg.readout_type == HCNNReadoutType::FLATTEN ? "FLATTEN" : "GAP")
              << " lr_max=" << args.lr_max
              << " num_outputs=" << cnn_cfg.num_outputs << "\n";

    // --- Phase 3: Stream through train_chars with online CNN updates. ---
    const auto pi = static_cast<float>(std::numbers::pi);
    const float lr_min = args.lr_max * args.lr_min_frac;
    const auto total_train_steps =
        static_cast<float>(args.train_chars) * args.num_passes;
    const std::size_t train_start_pos = corpus_pos;

    auto t_train_start = std::chrono::steady_clock::now();
    std::size_t global_step = 0;

    for (int pass = 0; pass < args.num_passes; ++pass) {
        corpus_pos = train_start_pos;

        for (std::size_t i = 0; i < args.train_chars; ++i) {
            BipolarBits(corpus.text[corpus_pos], step_bits);
            esn.Warmup(step_bits, 1);

            const float target = static_cast<float>(
                CharToClass(corpus, corpus.text[corpus_pos + 1]));

            float progress = static_cast<float>(global_step) / total_train_steps;
            float lr = lr_min + 0.5f * (args.lr_max - lr_min) *
                       (1.0f + std::cos(pi * progress));

            esn.TrainLiveStep(target, lr);
            ++corpus_pos;
            ++global_step;

            if (args.verbose && (global_step % 100000 == 0)) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - t_train_start).count();
                std::cerr << "[train] pass " << (pass + 1) << "/" << args.num_passes
                          << " step " << (i + 1) << "/" << args.train_chars
                          << " global=" << global_step
                          << " lr=" << lr
                          << " elapsed=" << elapsed << "s\n";
            }
        }

        std::cerr << "[train] pass " << (pass + 1) << "/" << args.num_passes
                  << " complete, global_steps=" << global_step << "\n";
    }

    corpus_pos = train_start_pos + args.train_chars;

    auto t_train_end = std::chrono::steady_clock::now();
    std::cerr << "[train] streaming training elapsed="
              << std::chrono::duration<double>(t_train_end - t_train_start).count()
              << "s (" << args.num_passes << " pass"
              << (args.num_passes > 1 ? "es" : "") << ")\n";

    // --- Phase 4: Stream through val_chars for evaluation. ---
    const std::size_t num_classes = kVocabSize;
    const std::size_t num_outputs = esn.NumOutputs();
    std::vector<float> logits(num_outputs);
    std::vector<std::size_t> sorted_idx(num_outputs);

    EvalMetrics val_m;
    val_m.per_class_correct.resize(num_classes, 0);
    val_m.per_class_total.resize(num_classes, 0);
    std::size_t correct1 = 0, correct3 = 0, correct5 = 0;
    double total_log_loss = 0.0;

    for (std::size_t i = 0; i < args.val_chars; ++i) {
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
            val_m.per_class_total[label]++;
            if (static_cast<int>(sorted_idx[0]) == label)
                val_m.per_class_correct[label]++;
        }

        ++corpus_pos;
    }

    val_m.top1 = static_cast<double>(correct1) / args.val_chars;
    val_m.top3 = static_cast<double>(correct3) / args.val_chars;
    val_m.top5 = static_cast<double>(correct5) / args.val_chars;
    val_m.bpc  = total_log_loss / (args.val_chars * std::log(2.0));

    PrintMetrics("streaming", "val", val_m, corpus, args.eval_worst_classes);

    // --- Text samples from val region. ---
    const std::size_t val_start = args.warmup_chars + args.warmup_train_chars + args.train_chars;
    const std::size_t prompt_len = std::min(args.eval_prompt_len, args.val_chars);
    for (std::size_t s = 0; s < args.eval_show_samples; ++s) {
        const std::size_t span = args.val_chars - prompt_len;
        const std::size_t offset = (args.eval_show_samples > 1)
            ? (s * span) / (args.eval_show_samples - 1) : 0;
        const std::size_t origin = val_start + offset;
        std::string prompt(corpus.text.data() + origin, prompt_len);
        std::string gen = GenerateText(esn, corpus, prompt, args.eval_gen_chars,
                                       args.eval_temperature,
                                       static_cast<unsigned>(gen_seed + s));
        std::cerr << "[streaming] sample " << (s + 1) << "/" << args.eval_show_samples
                  << " prompt=\"" << EscapeText(prompt) << "\"\n"
                  << "  -> \""    << EscapeText(gen)    << "\"\n";
    }

    // --- Phase 5: Save model. ---
    ModelFile mf;
    mf.dim                   = static_cast<std::uint32_t>(kDIM);
    mf.vocab                 = corpus.vocab;
    mf.meta.training_seed    = gen_seed;
    mf.meta.training_positions = static_cast<std::uint32_t>(args.train_chars);
    mf.meta.training_passes  = static_cast<std::uint32_t>(args.num_passes);
    mf.meta.git_sha          = args.git_sha;
    mf.reservoir_cfg         = esn.GetConfig();
    mf.cnn_cfg               = cnn_cfg;
    mf.readout               = ToSerial<kDIM>(esn.GetReadoutState());

    if (!SaveModelFile(args.output_path, mf)) {
        std::cerr << "error: failed to save model to " << args.output_path << "\n";
        return 4;
    }
    std::cerr << "[train] saved model to " << args.output_path << "\n";
    return 0;
}

}  // namespace hrccnn_lm_text
