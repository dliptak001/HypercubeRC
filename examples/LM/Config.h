#pragma once

/// @file Config.h
/// @brief Compile-time configuration for the LM character-level language model.
///
/// Edit these values, rebuild, and run — no command-line arguments needed.
/// kMode selects which pipeline stage to execute (Train, Eval, or Infer).

#include <cstddef>
#include <cstdint>
#include <string>

namespace lm::config {

inline constexpr std::size_t kDIM = 5;

enum class Mode { Train, Eval, Infer };

inline constexpr Mode kMode = Mode::Train;

// ---------------------------------------------------------------------------
//  Train — stream corpus through reservoir, train CNN readout, save model.
// ---------------------------------------------------------------------------
struct TrainCfg
{
    std::string   corpus_path      = "C:\\temp\\tinyshakespeare.txt";  // plain-text training file (ASCII)
    std::string   output_path      = "C:\\temp\\lm_v1.bin";           // where to write the trained model

    std::uint64_t gen_seed                 = 1235437745;   // RNG seed for sampling during eval
    bool          use_fixed_gen_seed       = true;         // false = draw a random seed at startup
    std::uint64_t reservoir_seed           = 34857575839839; // reservoir weight initialization seed
    bool          use_fixed_reservoir_seed = false;         // false = derive from gen_seed via hash

    std::size_t   warmup_chars       = 4096;    // chars streamed before any training (wash out transient)
    std::size_t   warmup_train_chars = 32768;   // chars used to compute per-vertex standardization stats
    std::size_t   train_chars        = 900000;  // chars per training pass
    int           num_passes         = 3;       // number of passes over the training region
    std::size_t   val_chars          = 50000;   // chars scored after each pass for eval metrics

    std::size_t   cascade_depth      = 5;       // number of stacked reservoir layers (1 = single, >1 = cascade)

    float         spectral_radius   = 0.90f;   // reservoir weight matrix scaling (controls echo memory)
    float         leak_rate         = 1.0f;     // 1.0 = full update; <1.0 = leaky integrator (smooths dynamics)
    float         input_scaling     = 0.02f;    // scale factor applied to input before injection into reservoir
    float         coupling_scaling  = 0.02f;    // inter-layer coupling weight scale (cascade depth > 1)
    int           coupling_mode     = 0;        // CouplingMode enum: 0=Raw 1=Binarize 2=Normalize 3=Center
    float         output_fraction   = 1.0f;     // fraction of reservoir vertices fed to the readout (0,1]

    int           cnn_num_layers    = 1;        // Conv+Pool pairs in the HCNN readout
    int           cnn_conv_channels = 8;        // base channel count (doubles per layer)

    int           mini_batch_size   = 512;      // states accumulated before each CNN weight update
    float         lr_max            = 0.0015f;  // peak learning rate (linearly decayed to floor)
    float         lr_floor_frac     = 0.5f;     // floor = lr_max * lr_floor_frac

    bool          verbose           = true;     // print progress every 100k steps
    std::size_t   eval_show_samples = 3;        // autoregressive text samples shown after final pass
    std::size_t   eval_prompt_len   = 64;       // chars of real text used to prime each sample
    std::size_t   eval_gen_chars    = 200;      // chars generated per sample
    float         eval_temperature  = 0.7f;     // sampling temperature (0 = greedy, higher = more random)
    std::size_t   eval_worst_classes = 5;       // number of worst-accuracy classes to report

    std::string   git_sha           = "";       // optional git SHA embedded in saved model for provenance
};

inline const TrainCfg kTrain;

// ---------------------------------------------------------------------------
//  Eval — load saved model, score teacher-forced accuracy on held-out text.
// ---------------------------------------------------------------------------
struct EvalCfg
{
    std::string   model_path       = "C:\\temp\\lm_v1.bin";            // trained model to evaluate
    std::string   corpus_path      = "C:\\temp\\tinyshakespeare.txt";  // corpus to score against
    std::size_t   warmup_chars     = 4096;       // chars streamed to warm up reservoir before scoring
    std::size_t   skip_chars       = 900000;   // chars to skip (e.g. training region) before eval
    std::size_t   eval_chars       = 100000;   // chars scored for metrics
    std::size_t   eval_worst_classes = 5;      // number of worst-accuracy classes to report
};

inline const EvalCfg kEval;

// ---------------------------------------------------------------------------
//  Infer — load saved model, autoregressively continue a prompt.
// ---------------------------------------------------------------------------
struct InferCfg
{
    std::string   model_path       = "C:\\temp\\lm_v1.bin";       // trained model to generate from
    std::string   prompt           = "To be, or not to be, ";  // text prefix fed to the reservoir
    std::size_t   num_chars        = 500;      // characters to generate after the prompt
    float         temperature      = 0.7f;     // sampling temperature (0 = greedy, higher = more random)
    std::uint64_t gen_seed         = 1235437745; // RNG seed for sampling reproducibility
};

inline const InferCfg kInfer;

}  // namespace lm::config
