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

EvalMetrics ComputeMetrics(ESN<kDIM>& esn,
                           const float* targets,
                           std::size_t start,
                           std::size_t count,
                           std::size_t num_classes)
{
    EvalMetrics m;
    m.per_class_correct.resize(num_classes, 0);
    m.per_class_total.resize(num_classes, 0);

    const std::size_t num_outputs = esn.NumOutputs();
    std::vector<float> logits(num_outputs);

    std::size_t correct1 = 0, correct3 = 0, correct5 = 0;
    double total_log_loss = 0.0;

    std::vector<std::size_t> sorted_idx(num_outputs);

    for (std::size_t i = 0; i < count; ++i) {
        esn.PredictRaw(start + i, logits.data());
        const int label = static_cast<int>(targets[start + i]);

        // Softmax for BPC.
        float max_logit = *std::max_element(logits.begin(),
                                            logits.begin() + static_cast<long>(num_outputs));
        double sum_exp = 0.0;
        for (std::size_t k = 0; k < num_outputs; ++k)
            sum_exp += std::exp(static_cast<double>(logits[k]) - max_logit);
        double log_prob = (logits[label] - max_logit) - std::log(sum_exp);
        total_log_loss -= log_prob;

        // Top-k: partial sort descending.
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
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

        // Per-class.
        if (label >= 0 && static_cast<std::size_t>(label) < num_classes) {
            m.per_class_total[label]++;
            if (static_cast<int>(sorted_idx[0]) == label)
                m.per_class_correct[label]++;
        }
    }

    m.top1 = static_cast<double>(correct1) / count;
    m.top3 = static_cast<double>(correct3) / count;
    m.top5 = static_cast<double>(correct5) / count;
    m.bpc  = total_log_loss / (count * std::log(2.0));
    return m;
}

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
        args.warmup_chars + args.train_chars + args.val_chars + 1;
    if (corpus.text.size() < total_chars) {
        std::cerr << "error: corpus has " << corpus.text.size()
                  << " chars, need " << total_chars
                  << " (warmup+train+val+1)\n";
        return 2;
    }
    const std::size_t N = (1ULL << kDIM);
    const std::size_t positions = args.train_chars + args.val_chars;
    const double states_gib =
        static_cast<double>(positions) * N * 4.0 / (1024.0 * 1024.0 * 1024.0);
    const double peak_gib = states_gib * 2.5;
    std::cerr << "[train] RAM estimate: positions=" << positions
              << " per-state=" << (N * 4 / 1024) << " KiB"
              << " states_buf=" << states_gib << " GiB"
              << " peak ~" << peak_gib << " GiB\n";
    if (peak_gib > 40.0) {
        std::cerr << "warning: estimated peak RAM > 40 GiB.  Shrink "
                     "train_chars / val_chars before running.\n";
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
    rcfg.seed            = reservoir_seed;
    rcfg.num_inputs      = kInputBits;
    rcfg.output_fraction = args.output_fraction;

    std::cerr << "[train] DIM=" << kDIM << " N=" << N
              << " num_inputs=" << rcfg.num_inputs
              << " output_fraction=" << rcfg.output_fraction
              << " reservoir_seed=" << rcfg.seed << "\n";
    std::cerr << "[train] warmup_chars=" << args.warmup_chars
              << " train_chars=" << args.train_chars
              << " val_chars=" << args.val_chars
              << " epochs=" << args.epochs
              << " batch_size=" << args.batch_size << "\n";

    ESN<kDIM> esn(rcfg, ReadoutType::HCNN, FeatureMode::Raw);

    std::vector<float> bits(total_chars * kInputBits);
    for (std::size_t t = 0; t < total_chars; ++t)
        BipolarBits(corpus.text[t], bits.data() + t * kInputBits);

    auto t_start = std::chrono::steady_clock::now();

    esn.ResetReservoirOnly();
    if (args.warmup_chars > 0)
        esn.Warmup(bits.data(), args.warmup_chars);

    const std::size_t run_offset = args.warmup_chars;
    esn.Run(bits.data() + run_offset * kInputBits,
            args.train_chars + args.val_chars);

    const std::size_t train_positions = args.train_chars;
    const std::size_t val_positions   = args.val_chars;

    std::vector<float> targets(train_positions + val_positions);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        targets[i] = static_cast<float>(
            CharToClass(corpus, corpus.text[run_offset + i + 1]));
    }

    auto t_collected = std::chrono::steady_clock::now();
    std::cerr << "[train] collection: train_positions=" << train_positions
              << " val_positions=" << val_positions
              << " elapsed="
              << std::chrono::duration<double>(t_collected - t_start).count()
              << "s\n";

    CNNReadoutConfig cnn_cfg = hcnn_presets::HRCCNNBaseline<kDIM>();
    cnn_cfg.task              = HCNNTask::Classification;
    cnn_cfg.num_outputs       = static_cast<int>(kVocabSize);
    cnn_cfg.num_layers        = args.cnn_num_layers;
    cnn_cfg.conv_channels     = args.cnn_conv_channels;
    cnn_cfg.epochs            = args.epochs;
    cnn_cfg.batch_size        = args.batch_size;
    cnn_cfg.verbose           = args.verbose;
    cnn_cfg.verbose_train_acc = args.verbose_train_acc;
    cnn_cfg.lr_decay_epochs   = args.lr_decay_epochs;

    std::cerr << "[train] CNN cfg: nl=" << cnn_cfg.num_layers
              << " ch=" << cnn_cfg.conv_channels
              << " head=" << (cnn_cfg.readout_type == HCNNReadoutType::FLATTEN ? "FLATTEN" : "GAP")
              << " lr=" << cnn_cfg.lr_max
              << " epochs=" << cnn_cfg.epochs
              << " bs=" << cnn_cfg.batch_size
              << " num_outputs=" << cnn_cfg.num_outputs << "\n";

    // --- Early stopping state. ---
    double best_val_metric = -1.0;
    int    evals_without_improvement = 0;

    // --- Eval reporter. ---
    const std::size_t num_classes = kVocabSize;

    auto run_eval_report = [&](const std::string& tag) {
        const double train_top1 = esn.Accuracy(targets.data(), 0, train_positions);
        std::cerr << "[" << tag << "] train: top1=" << train_top1 << "\n";

        double val_top1 = 0.0;
        if (val_positions > 0) {
            EvalMetrics m_val = ComputeMetrics(esn, targets.data(),
                                               train_positions, val_positions,
                                               num_classes);
            PrintMetrics(tag, "val", m_val, corpus, args.eval_worst_classes);
            val_top1 = m_val.top1;
        }

        const std::size_t prompt_len = std::min(args.eval_prompt_len, args.val_chars);
        const std::size_t val_start  = args.warmup_chars + args.train_chars;
        const std::size_t max_samples = args.eval_show_samples;
        for (std::size_t s = 0; s < max_samples; ++s) {
            const std::size_t span = args.val_chars - prompt_len;
            const std::size_t offset = (max_samples > 1)
                ? (s * span) / (max_samples - 1)
                : 0;
            const std::size_t origin = val_start + offset;
            std::string prompt(corpus.text.data() + origin, prompt_len);
            std::string gen = GenerateText(esn, corpus, prompt, args.eval_gen_chars,
                                           args.eval_temperature,
                                           static_cast<unsigned>(gen_seed + s));
            std::cerr << "[" << tag << "] sample " << (s + 1) << "/" << max_samples
                      << " prompt=\"" << EscapeText(prompt) << "\"\n"
                      << "  -> \""    << EscapeText(gen)    << "\"\n";
        }

        return val_top1;
    };

    // --- Train + checkpoint hook. ---
    CNNTrainHooks hooks;
    hooks.eval_every_epochs = args.eval_every_epochs;
    hooks.epoch_callback = [&](int epoch_done, int total_epochs, float lr) {
        std::ostringstream tag;
        tag << "eval e=" << epoch_done << "/" << total_epochs << " lr=" << lr;
        double val_top1 = run_eval_report(tag.str());

        // Early stopping.
        if (args.eval_patience > 0 && val_positions > 0) {
            if (val_top1 > best_val_metric) {
                best_val_metric = val_top1;
                evals_without_improvement = 0;
            } else {
                ++evals_without_improvement;
                if (evals_without_improvement >= args.eval_patience) {
                    std::cerr << "[train] early stop: val top1 has not improved for "
                              << evals_without_improvement << " evals\n";
                    hooks.stop_requested = true;
                }
            }
        }

        ModelFile ckpt;
        ckpt.dim                  = static_cast<std::uint32_t>(kDIM);
        ckpt.vocab                = corpus.vocab;
        ckpt.meta.training_seed   = gen_seed;
        ckpt.meta.training_positions = static_cast<std::uint32_t>(args.train_chars);
        ckpt.meta.training_epochs = static_cast<std::uint32_t>(epoch_done);
        ckpt.meta.git_sha         = args.git_sha;
        ckpt.reservoir_cfg        = esn.GetConfig();
        ckpt.cnn_cfg              = cnn_cfg;
        ckpt.readout              = ToSerial<kDIM>(esn.GetReadoutState());

        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), ".e%04d.bin", epoch_done);
        const std::string ckpt_path = args.output_path + suffix;
        if (SaveModelFile(ckpt_path, ckpt))
            std::cerr << "[train] checkpoint saved: " << ckpt_path << "\n";
        else
            std::cerr << "warning: failed to save checkpoint to " << ckpt_path << "\n";
    };

    auto t_train_start = std::chrono::steady_clock::now();
    esn.Train(targets.data(), train_positions, cnn_cfg, hooks);
    auto t_train_end = std::chrono::steady_clock::now();
    std::cerr << "[train] training elapsed="
              << std::chrono::duration<double>(t_train_end - t_train_start).count()
              << "s\n";

    if (hooks.stop_requested)
        std::cerr << "[train] stopped early at best_val_top1=" << best_val_metric << "\n";

    if (hooks.eval_every_epochs <= 0 || cnn_cfg.epochs <= 0)
        run_eval_report("train-final");

    ModelFile mf;
    mf.dim                   = static_cast<std::uint32_t>(kDIM);
    mf.vocab                 = corpus.vocab;
    mf.meta.training_seed    = gen_seed;
    mf.meta.training_positions = static_cast<std::uint32_t>(args.train_chars);
    mf.meta.training_epochs  = static_cast<std::uint32_t>(args.epochs);
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
