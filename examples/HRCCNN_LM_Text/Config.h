#pragma once

// Hardcoded run-time configuration for HRCCNN_LM_Text.  Edit these values,
// rebuild, and launch the exe with no arguments.  kMode selects which
// Run* gets invoked; each mode reads its own struct below.
//
// NOTE: corpus_path must point to a plain-text file (e.g. tinyshakespeare
// at ~1 MB from https://raw.githubusercontent.com/karpathy/char-rnn/
// master/data/tinyshakespeare/input.txt).  First run builds the vocab
// from that file and embeds it into the saved model, so eval/infer on
// a given checkpoint need not see the corpus again.

#include <cstddef>
#include <cstdint>
#include <string>

namespace hrccnn_lm_text::config {

enum class Mode { Train, Eval, Infer };

// -----------------------------------------------------------------------------
// Pick ONE:
// -----------------------------------------------------------------------------
inline constexpr Mode kMode = Mode::Train;

// -----------------------------------------------------------------------------
// Train — chunk the corpus, teacher-force, train CNN readout, save model.
// -----------------------------------------------------------------------------
struct TrainCfg
{
    std::string   corpus_path      = "C:\\temp\\tinyshakespeare.txt";
    std::string   output_path      = "C:\\temp\\text_v1.bin";

    // Seeds.
    std::uint64_t gen_seed                 = 12345;   ///< chunk-start sampling
    bool          use_fixed_gen_seed       = true;
    std::uint64_t reservoir_seed           = 0;
    bool          use_fixed_reservoir_seed = false;   ///< false → derive from gen_seed

    // Streaming scheme: load the corpus once, drive the reservoir through
    // one continuous span of warmup + train + val characters.  ONE state
    // per character, nothing duplicated.
    //
    // Peak RAM dominated by:
    //   esn.States()  = (train_chars + val_chars) * N * 4 bytes
    //   CNN std copy  = same again during training
    //   HCNNStates    = transient, half that, per eval call
    //
    // At DIM 12 (N=4096, 16 KiB/state) the defaults below give
    // 220 k positions × 16 KiB × ~2.5 = ~9 GiB peak.  Fits in 16 GiB.
    std::size_t   warmup_chars     = 64;      ///< transient clear before collection
    std::size_t   train_chars      = 200000;  ///< positions collected for training
    std::size_t   val_chars        = 20000;   ///< positions collected for validation

    // CNN training.
    int           epochs            = 100;
    int           batch_size        = 4096;
    float         output_fraction   = 0.5f;
    int           cnn_num_layers    = 2;
    int           cnn_conv_channels = 16;
    int           lr_decay_epochs   = 200;   ///< cosine decay horizon; 0 = collapse to epochs

    // Verbosity / hooks.
    bool          verbose           = true;
    bool          verbose_train_acc = false;  ///< per-epoch full-train forward pass (pricey)
    int           eval_every_epochs = 25;
    std::size_t   eval_show_samples = 3;      ///< autoreg text samples printed per eval
    std::size_t   eval_prompt_len   = 64;     ///< chars from val region used as prompt
    std::size_t   eval_gen_chars    = 200;    ///< chars generated per sample
    float         eval_temperature  = 0.8f;   ///< sampling temp for autoreg (0 = greedy)
    int           eval_patience     = 0;      ///< stop after N evals with no val improvement; 0 = off
    std::size_t   eval_worst_classes = 5;     ///< per-class confusion: show N worst

    std::string   git_sha           = "";
};

inline const TrainCfg kTrain;

// -----------------------------------------------------------------------------
// Eval — reload saved model, score teacher-forced char accuracy on held-out chunks.
// -----------------------------------------------------------------------------
struct EvalCfg
{
    std::string   model_path       = "C:\\temp\\text_v1.bin";
    std::string   corpus_path      = "C:\\temp\\tinyshakespeare.txt";
    // Must match train's warmup+train_chars offset to score the same val
    // region — corpus is deterministic and positional.
    std::size_t   warmup_chars     = 64;
    std::size_t   skip_chars       = 200000;  ///< chars to drive past before scoring
    std::size_t   eval_chars       = 20000;   ///< positions scored
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
    unsigned      seed             = 42;
};

inline const InferCfg kInfer;

}  // namespace hrccnn_lm_text::config
