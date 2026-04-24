#include "Trainer.h"
#include "Generator.h"
#include "Presets.h"

#include <chrono>
#include <iostream>
#include <random>

namespace lm {

// =========================================================================
//  Construction
// =========================================================================

template <std::size_t DIM>
Trainer<DIM>::Trainer(Model<DIM>& model,
                      const Corpus& corpus,
                      const Vocabulary& vocab,
                      const config::TrainCfg& cfg)
    : model_(model), corpus_(corpus), vocab_(vocab), cfg_(cfg)
{
    // Resolve the generation seed.
    if (cfg_.use_fixed_gen_seed) {
        gen_seed_ = cfg_.gen_seed;
    } else {
        std::random_device rd;
        gen_seed_ = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }
}

// =========================================================================
//  Public interface
// =========================================================================

template <std::size_t DIM>
int Trainer<DIM>::Run()
{
    // Validate corpus size.
    const std::size_t train_total =
        cfg_.warmup_chars + cfg_.warmup_train_chars + cfg_.train_chars + 1;
    const bool will_wrap_eval =
        (train_total + cfg_.val_chars) > corpus_.Size();
    const std::size_t min_corpus = will_wrap_eval
        ? std::max(train_total, cfg_.val_chars + 1)
        : (train_total + cfg_.val_chars);

    if (corpus_.Size() < min_corpus) {
        std::cerr << "[train] error: corpus has " << corpus_.Size()
                  << " chars, need " << min_corpus << "\n";
        return 2;
    }

    std::cerr << "[train] corpus=" << corpus_.Size() << " chars"
              << "  vocab=" << vocab_.Tokens().size() << " tokens\n";

    WarmupReservoir();
    CollectStandardization();
    StreamTrain();
    SaveModel();

    return 0;
}

// =========================================================================
//  Phase 1: Reservoir warmup
// =========================================================================

template <std::size_t DIM>
void Trainer<DIM>::WarmupReservoir()
{
    model_.Reset();
    for (std::size_t i = 0; i < cfg_.warmup_chars; ++i)
        model_.Step(corpus_.At(corpus_pos_++));

    std::cerr << "[train] warmup: " << cfg_.warmup_chars << " chars\n";
}

// =========================================================================
//  Phase 2: Standardization collection
// =========================================================================

template <std::size_t DIM>
void Trainer<DIM>::CollectStandardization()
{
    std::string warmup_text(
        corpus_.Text().data() + corpus_pos_,
        cfg_.warmup_train_chars);

    ReadoutArchConfig arch = presets::Baseline<DIM>().arch;
    arch.task          = ReadoutTask::Classification;
    arch.num_outputs   = static_cast<int>(Vocabulary::kSize);
    arch.num_layers    = cfg_.cnn_num_layers;
    arch.conv_channels = cfg_.cnn_conv_channels;

    model_.InitReadout(warmup_text, arch);
    corpus_pos_ += cfg_.warmup_train_chars;

    std::cerr << "[train] standardization: " << cfg_.warmup_train_chars
              << " chars  CNN: nl=" << arch.num_layers
              << " ch=" << arch.conv_channels
              << " outputs=" << arch.num_outputs << "\n";
}

// =========================================================================
//  Phase 3: Streaming mini-batch training
// =========================================================================

template <std::size_t DIM>
void Trainer<DIM>::StreamTrain()
{
    train_start_pos_ = corpus_pos_;
    const int K = cfg_.mini_batch_size;
    const std::size_t state_dim = model_.NumOutputVerts();

    // Allocate accumulation buffers once.
    accum_states_.resize(K * state_dim);
    accum_targets_.resize(K);
    accum_count_ = 0;

    // LR schedule: linear decay across all passes.
    total_batches_ =
        (cfg_.train_chars * cfg_.num_passes + K - 1) / K;
    const float lr_min = cfg_.lr_max * cfg_.lr_floor_frac;
    batch_index_ = 0;
    current_lr_ = cfg_.lr_max;

    std::cerr << "[train] streaming: " << cfg_.train_chars << " chars x "
              << cfg_.num_passes << " passes"
              << "  batch=" << K
              << "  lr=" << cfg_.lr_max << " -> " << lr_min << "\n";

    auto t_start = std::chrono::steady_clock::now();
    std::size_t global_step = 0;

    for (int pass = 0; pass < cfg_.num_passes; ++pass) {
        corpus_pos_ = train_start_pos_;

        for (std::size_t i = 0; i < cfg_.train_chars; ++i) {
            // Advance reservoir one character.
            model_.Step(corpus_.At(corpus_pos_));

            // Accumulate state + target for the mini-batch.
            model_.CopyLiveState(
                accum_states_.data() + accum_count_ * state_dim);
            accum_targets_[accum_count_] =
                vocab_.CharToClass(corpus_.At(corpus_pos_ + 1));
            ++accum_count_;
            ++corpus_pos_;
            ++global_step;

            // Flush when the batch is full.
            if (accum_count_ == K)
                FlushBatch();

            if (cfg_.verbose && (global_step % 100000 == 0)) {
                auto now = std::chrono::steady_clock::now();
                double elapsed =
                    std::chrono::duration<double>(now - t_start).count();
                std::cerr << "[train] pass " << (pass + 1) << "/" << cfg_.num_passes
                          << " step " << (i + 1) << "/" << cfg_.train_chars
                          << " lr=" << current_lr_
                          << " elapsed=" << elapsed << "s\n";
            }
        }

        // Flush any remaining samples at end of pass.
        if (accum_count_ > 0)
            FlushBatch();

        // Mid-training evaluation.
        const std::size_t default_eval_start = train_start_pos_ + cfg_.train_chars;
        const bool wrap = (default_eval_start + cfg_.val_chars + 1) > corpus_.Size();
        const std::size_t eval_start = wrap ? 0 : default_eval_start;
        const bool show_samples = (pass == cfg_.num_passes - 1);

        std::string tag = "pass " + std::to_string(pass + 1)
                        + "/" + std::to_string(cfg_.num_passes);
        Evaluate(tag, eval_start, show_samples);
    }

    auto t_end = std::chrono::steady_clock::now();
    std::cerr << "[train] training complete: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << "s\n";
}

template <std::size_t DIM>
void Trainer<DIM>::FlushBatch()
{
    float frac = static_cast<float>(batch_index_)
               / static_cast<float>(total_batches_);
    float lr_min = cfg_.lr_max * cfg_.lr_floor_frac;
    current_lr_ = cfg_.lr_max - (cfg_.lr_max - lr_min) * frac;

    model_.TrainBatch(accum_states_.data(), accum_targets_.data(),
                      accum_count_, current_lr_);
    ++batch_index_;
    accum_count_ = 0;
}

// =========================================================================
//  Phase 4: Evaluation
// =========================================================================

template <std::size_t DIM>
void Trainer<DIM>::Evaluate(const std::string& tag,
                            std::size_t eval_start,
                            bool show_samples)
{
    // Save reservoir state so training can resume after eval.
    std::vector<float> saved_state, saved_output;
    model_.SaveState(saved_state, saved_output);

    // Stream through the validation region.
    Evaluator eval(Vocabulary::kSize);
    std::vector<float> logits(model_.NumOutputs());
    std::size_t eval_pos = eval_start;

    for (std::size_t i = 0; i < cfg_.val_chars; ++i) {
        model_.Step(corpus_.At(eval_pos));
        model_.Predict(logits.data());

        int label = vocab_.CharToClass(corpus_.At(eval_pos + 1));
        eval.Record(logits.data(), label);
        ++eval_pos;
    }

    // Report metrics.
    Metrics m = eval.Compute();
    std::cerr << Evaluator::FormatMetrics(tag, m) << "\n";

    auto worst = eval.WorstClasses(cfg_.eval_worst_classes);
    if (!worst.empty())
        std::cerr << Evaluator::FormatWorstClasses(tag, worst, vocab_) << "\n";

    // Autoregressive text samples (final pass only).
    if (show_samples && cfg_.eval_show_samples > 0) {
        const std::size_t prompt_len =
            std::min(cfg_.eval_prompt_len, cfg_.val_chars);

        for (std::size_t s = 0; s < cfg_.eval_show_samples; ++s) {
            const std::size_t span = cfg_.val_chars - prompt_len;
            const std::size_t offset = (cfg_.eval_show_samples > 1)
                ? (s * span) / (cfg_.eval_show_samples - 1) : 0;
            const std::size_t origin = eval_start + offset;

            std::string prompt(corpus_.Text().data() + origin, prompt_len);

            Generator<DIM> gen(model_, vocab_);
            std::string text = gen.Generate(
                prompt, cfg_.eval_gen_chars,
                cfg_.eval_temperature,
                static_cast<unsigned>(gen_seed_ + s));

            std::cerr << "[sample " << (s + 1) << "/" << cfg_.eval_show_samples
                      << "] \"" << text << "\"\n";
        }
    }

    // Restore reservoir state so training can continue.
    model_.RestoreState(saved_state, saved_output);
}

// =========================================================================
//  Phase 5: Save model
// =========================================================================

template <std::size_t DIM>
void Trainer<DIM>::SaveModel()
{
    if (cfg_.output_path.empty()) return;

    TrainingMetadata meta;
    meta.training_seed      = gen_seed_;
    meta.training_positions = static_cast<std::uint32_t>(cfg_.train_chars);
    meta.training_passes    = static_cast<std::uint32_t>(cfg_.num_passes);
    meta.git_sha            = cfg_.git_sha;

    model_.Save(cfg_.output_path, vocab_, meta);
    std::cerr << "[train] saved model to " << cfg_.output_path << "\n";
}

// =========================================================================
//  Explicit instantiation
// =========================================================================

using namespace lm::config;
template class Trainer<kDIM>;

}  // namespace lm
