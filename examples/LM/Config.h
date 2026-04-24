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

inline constexpr std::size_t kDIM = 13;

enum class Mode { Train, Eval, Infer };

inline constexpr Mode kMode = Mode::Train;

// ---------------------------------------------------------------------------
//  Train — stream corpus through reservoir, train CNN readout, save model.
// ---------------------------------------------------------------------------
struct TrainCfg
{
    std::string   corpus_path      = "C:\\temp\\tinyshakespeare.txt";
    std::string   output_path      = "C:\\temp\\lm_v1.bin";

    std::uint64_t gen_seed                 = 1235437745;
    bool          use_fixed_gen_seed       = true;
    std::uint64_t reservoir_seed           = 34857575839839;
    bool          use_fixed_reservoir_seed = false;

    std::size_t   warmup_chars       = 64;
    std::size_t   warmup_train_chars = 32768;
    std::size_t   train_chars        = 900000;
    int           num_passes         = 3;
    std::size_t   val_chars          = 100000;

    float         spectral_radius   = 0.90f;
    float         leak_rate         = 1.0f;
    float         output_fraction   = 0.5f;

    int           cnn_num_layers    = 1;
    int           cnn_conv_channels = 4;

    int           mini_batch_size   = 512;
    float         lr_max            = 0.0015f;
    float         lr_floor_frac     = 0.5f;

    bool          verbose           = true;
    std::size_t   eval_show_samples = 3;
    std::size_t   eval_prompt_len   = 64;
    std::size_t   eval_gen_chars    = 200;
    float         eval_temperature  = 0.8f;
    std::size_t   eval_worst_classes = 5;

    std::string   git_sha           = "";
};

inline const TrainCfg kTrain;

// ---------------------------------------------------------------------------
//  Eval — load saved model, score teacher-forced accuracy on held-out text.
// ---------------------------------------------------------------------------
struct EvalCfg
{
    std::string   model_path       = "C:\\temp\\lm_v1.bin";
    std::string   corpus_path      = "C:\\temp\\tinyshakespeare.txt";
    std::size_t   warmup_chars     = 64;
    std::size_t   skip_chars       = 900000;
    std::size_t   eval_chars       = 100000;
    std::size_t   eval_worst_classes = 5;
};

inline const EvalCfg kEval;

// ---------------------------------------------------------------------------
//  Infer — load saved model, autoregressively continue a prompt.
// ---------------------------------------------------------------------------
struct InferCfg
{
    std::string   model_path       = "C:\\temp\\lm_v1.bin";
    std::string   prompt           = "To be, or not to be, ";
    std::size_t   num_chars        = 500;
    float         temperature      = 0.8f;
    unsigned      gen_seed         = 1235437745;
};

inline const InferCfg kInfer;

}  // namespace lm::config
