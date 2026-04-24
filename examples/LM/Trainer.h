#pragma once

/// @file Trainer.h
/// @brief Streaming training pipeline for the character-level language model.
///
/// The Trainer orchestrates the full training lifecycle:
///   1. Reservoir warmup (wash out zero-init transient)
///   2. Standardization collection (compute per-vertex mean/std for the CNN)
///   3. Streaming mini-batch training with linear LR decay
///   4. Periodic evaluation via an Evaluator
///   5. Model saving
///
/// The reservoir advances one character at a time (sequential — state at T
/// depends on T-1).  CNN weight updates happen every mini_batch_size steps,
/// using accumulated state snapshots dispatched as a parallel TrainBatch call.

#include <cstddef>
#include <string>
#include <vector>

#include "Config.h"
#include "Corpus.h"
#include "Evaluator.h"
#include "Model.h"
#include "Vocabulary.h"

namespace lm {

template <std::size_t DIM>
class Trainer
{
public:
    /// Construct a trainer for the given model, corpus, and vocabulary.
    /// The Trainer does not own these objects — they must outlive it.
    Trainer(Model<DIM>& model,
            const Corpus& corpus,
            const Vocabulary& vocab,
            const config::TrainCfg& cfg);

    /// Execute the full training pipeline.
    /// Returns 0 on success, non-zero on error.
    int Run();

private:
    // --- Pipeline phases ---
    void WarmupReservoir();
    void CollectStandardization();
    void StreamTrain();
    void Evaluate(const std::string& tag, std::size_t eval_start, bool show_samples);
    void SaveModel();

    // --- Helpers ---
    void FlushBatch();

    Model<DIM>&             model_;
    const Corpus&           corpus_;
    const Vocabulary&       vocab_;
    const config::TrainCfg& cfg_;

    // Corpus position tracking.
    std::size_t corpus_pos_       = 0;
    std::size_t train_start_pos_  = 0;

    // Mini-batch accumulation buffers (allocated once, reused every batch).
    std::vector<float> accum_states_;
    std::vector<int>   accum_targets_;
    int                accum_count_ = 0;

    // LR schedule state.
    std::size_t total_batches_ = 0;
    std::size_t batch_index_   = 0;
    float       current_lr_    = 0.0f;

    // Seed for reproducibility.
    std::uint64_t gen_seed_ = 0;
};

}  // namespace lm
