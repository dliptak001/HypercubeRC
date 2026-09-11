#pragma once

// Hardcoded run-time configuration for LM_Text.  Edit these values,
// rebuild, and launch the exe with no arguments.  kMode selects which
// Run* gets invoked; each mode reads its own struct below.
//
// NOTE: corpus_path must point to a plain-text file (e.g. tinyshakespeare
// at ~1 MB from https://raw.githubusercontent.com/karpathy/char-rnn/
// master/data/tinyshakespeare/input.txt).  All bytes must fall within
// the fixed 96-token vocabulary (newline + printable ASCII 0x20-0x7E).
// The fixed vocab is embedded into saved models for verification.

#include <cstddef>
#include <cstdint>
#include <string>

namespace lm_text::config {

inline constexpr std::size_t kDIM = 13;

enum class Mode { Train, Eval, Infer };

// -----------------------------------------------------------------------------
// Pick ONE:
// -----------------------------------------------------------------------------
inline constexpr Mode kMode = Mode::Train;

// -----------------------------------------------------------------------------
// Train — stream the corpus, teacher-force, train CNN readout, save model.
// -----------------------------------------------------------------------------
struct TrainCfg
{
    std::string   corpus_path      = "C:\\temp\\tinyshakespeare.txt";
    std::string   output_path      = "C:\\temp\\text_v1.bin";

    // Seeds.
    std::uint64_t gen_seed                 = 1235437745;   ///< master seed
    bool          use_fixed_gen_seed       = true;
    std::uint64_t reservoir_seed           = 34857575839839;
    bool          use_fixed_reservoir_seed = false;   ///< false → derive from gen_seed

    // Streaming training: drive the reservoir one char at a time,
    // update the CNN readout after each step.  No states buffer needed.
    // Peak RAM: reservoir (N*4 bytes) + CNN weights — under 1 GiB.
    std::size_t   warmup_chars       = 64;      ///< transient warmup before training
    std::size_t   warmup_train_chars = 32768;   ///< states collected for CNN standardization
    std::size_t   train_chars        = 900000;  ///< chars streamed for online CNN training
    int           num_passes         = 3;        ///< corpus passes (reservoir continues, no reset)
    std::size_t   val_chars          = 100000;  ///< chars streamed for evaluation

    // Reservoir dynamics.
    float         spectral_radius   = 0.90f;
    float         leak_rate         = 1.0f;    ///< 1.0 = full replacement, <1.0 = leaky integrator
    float         output_fraction   = 0.5f;

    // CNN architecture + training.
    int           cnn_num_layers    = 1;
    int           cnn_conv_channels = 4;
    int           eval_every_chars  = 100000;  ///< streaming eval interval (0 = end only)

    // Gradient accumulation: reservoir streams one char at a time,
    // but CNN weight updates happen every mini_batch_size steps.
    // Parallelized across threads via HCNN::TrainBatch.
    int           mini_batch_size   = 512;

    // LR schedule: linear decay from lr_max to lr_max * lr_floor_frac
    // across all mini-batch updates (all passes combined, no reset).
    float         lr_max            = 0.0015f;
    float         lr_floor_frac     = 0.5f;    ///< lr decays linearly to lr_max * lr_floor_frac

    // Verbosity / eval.
    bool          verbose           = true;
    std::size_t   eval_show_samples = 3;      ///< autoreg text samples printed per eval
    std::size_t   eval_prompt_len   = 64;     ///< chars from val region used as prompt
    std::size_t   eval_gen_chars    = 200;    ///< chars generated per sample
    float         eval_temperature  = 0.8f;   ///< sampling temp for autoreg (0 = greedy)
    std::size_t   eval_worst_classes = 5;     ///< per-class confusion: show N worst

    std::string   git_sha           = "";
};

inline const TrainCfg kTrain;

// -----------------------------------------------------------------------------
// Eval — reload saved model, stream through corpus, score teacher-forced accuracy.
// -----------------------------------------------------------------------------
struct EvalCfg
{
    std::string   model_path       = "C:\\temp\\text_v1.bin";
    std::string   corpus_path      = "C:\\temp\\tinyshakespeare.txt";
    // Streaming eval: drive reservoir one char at a time, predict at each step.
    // warmup_chars + skip_chars positions corpus_pos to the eval region.
    std::size_t   warmup_chars     = 64;
    std::size_t   skip_chars       = 900000;  ///< chars to stream past before scoring
    std::size_t   eval_chars       = 100000;  ///< positions scored
    std::size_t   eval_worst_classes = 5;     ///< per-class confusion: show N worst
};

inline const EvalCfg kEval;

// -----------------------------------------------------------------------------
// Infer — load saved model and autoregressively continue a prompt.
// -----------------------------------------------------------------------------
struct InferCfg
{
    std::string   model_path       = "C:\\temp\\text_v1.bin";
    std::string   prompt           = "To be, or not to be, ";
    std::size_t   num_chars        = 500;
    float         temperature      = 0.8f;   ///< 0 = greedy argmax
    unsigned      gen_seed         = 1235437745;  ///< sampling RNG seed
};

inline const InferCfg kInfer;

}  // namespace lm_text::config
