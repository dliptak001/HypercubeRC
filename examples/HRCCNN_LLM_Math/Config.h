#pragma once

// Hardcoded run-time configuration for HRCCNN_LLM_Math.  Edit these values,
// rebuild, and launch the exe with no arguments.  kMode selects which Run*
// gets invoked; each mode reads its own struct below.

#include <cstddef>
#include <cstdint>
#include <string>

namespace hrccnn_llm_math::config {

enum class Mode { Generate, Train, Eval, Infer };

// -----------------------------------------------------------------------------
// Pick ONE:
// -----------------------------------------------------------------------------
inline constexpr Mode kMode = Mode::Train;

// -----------------------------------------------------------------------------
// Generate — sample expression lines from the grammar and verify them.
// -----------------------------------------------------------------------------
struct GenerateCfg
{
    std::size_t   samples        = 1000;
    std::uint64_t seed           = 12345;
    bool          use_fixed_seed = true;    ///< false → draw from std::random_device
    bool          rhs_filter_999 = true;
    bool          verify         = true;
    bool          quiet          = false;   ///< true → suppress per-line printing
};

inline const GenerateCfg kGenerate;

// -----------------------------------------------------------------------------
// Train — teacher-force, train CNN readout, save model, run autoreg sanity.
// -----------------------------------------------------------------------------
struct TrainCfg
{
    std::string   output_path        = "C:\\temp\\math_v1.bin";
    std::size_t   samples            = 15000;
    std::size_t   val_samples        = 2000;
    std::uint64_t gen_seed           = 12345;
    bool          use_fixed_gen_seed = true;     ///< false → random
    std::uint64_t reservoir_seed     = 0;        ///< 0 + !use_fixed → derived from gen_seed
    bool          use_fixed_reservoir_seed = false;
    int           epochs             = 100;
    int           batch_size         = 4096;
    float         output_fraction    = 0.5f;
    std::size_t   autoreg_samples    = 64;       ///< val lines to autoregressively score
    std::string   git_sha            = "";
    bool          verbose            = true;
    bool          rhs_filter_999     = true;

    // CNN capacity overrides (applied on top of HRCCNNBaseline<DIM>). The
    // baseline is nl=1/ch=8 — too small for arithmetic; nl=2/ch=16 is the
    // first capacity-bump probe.
    int           cnn_num_layers     = 2;
    int           cnn_conv_channels  = 16;

    // Mid-training eval cadence. 0 disables periodic eval; >0 prints a full
    // teacher-forced + autoregressive report every N completed epochs.
    int           eval_every_epochs  = 25;
    std::size_t   eval_show_samples  = 25;

    // Cosine LR decay horizon (epochs). Lets us shorten a run for wall-clock
    // (epochs=100) without compressing the decay; training stops after
    // `epochs` but lr traces only the first `epochs/lr_decay_epochs` of the
    // curve. Set to 0 to collapse back to `epochs`.
    int           lr_decay_epochs    = 200;

    // Per-epoch training-accuracy logging. Pricey — a full forward pass over
    // the training set every epoch. The eval hook already reports accuracy
    // every `eval_every_epochs`, so leave this off unless you truly need
    // every-epoch acc.
    bool          verbose_train_acc  = false;

    // Line-generation cache.  Hits disk at {output_path}.lines.bin; lets
    // subsequent runs skip the expensive generator rejection-sampling loop
    // when gen_seed / samples / val_samples / rhs_filter_999 match.
    bool          use_line_cache     = true;
};

inline const TrainCfg kTrain;

// -----------------------------------------------------------------------------
// Eval — reload saved model, score on fresh held-out set.
// -----------------------------------------------------------------------------
struct EvalCfg
{
    std::string   model_path         = "C:\\temp\\math_v1.bin";
    std::size_t   samples            = 2000;
    std::uint64_t seed               = 54321;
    bool          use_fixed_seed     = true;
    bool          skip_char_accuracy = false;
};

inline const EvalCfg kEval;

// -----------------------------------------------------------------------------
// Infer — load saved model and autoregressively complete an LHS.
// -----------------------------------------------------------------------------
struct InferCfg
{
    std::string   model_path       = "C:\\temp\\math_v1.bin";
    std::string   input            = "123+456";
    std::size_t   max_output_chars = 16;
};

inline const InferCfg kInfer;

}  // namespace hrccnn_llm_math::config
