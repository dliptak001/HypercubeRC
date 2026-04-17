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
    std::size_t   samples            = 5000;
    std::size_t   val_samples        = 2000;
    std::uint64_t gen_seed           = 12345;
    bool          use_fixed_gen_seed = true;     ///< false → random
    std::uint64_t reservoir_seed     = 0;        ///< 0 + !use_fixed → derived from gen_seed
    bool          use_fixed_reservoir_seed = false;
    int           epochs             = 1000;
    int           batch_size         = 4096;
    float         output_fraction    = 0.125f;
    std::size_t   autoreg_samples    = 64;       ///< val lines to autoregressively score
    std::string   git_sha            = "";
    bool          verbose            = true;
    bool          rhs_filter_999     = true;
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
