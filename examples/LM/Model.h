#pragma once

/// @file Model.h
/// @brief Character-level language model wrapping an ESN reservoir + CNN readout.
///
/// The Model class owns the ESN pipeline and provides a character-level
/// interface: step with a character, predict the next character's class
/// distribution, and save/load trained models to disk.
///
/// Template parameter DIM is the hypercube dimension (N = 2^DIM neurons).

#include <cstdint>
#include <string>
#include <vector>

#include "ESN.h"
#include "Vocabulary.h"

namespace lm {

/// Training metadata embedded in saved model files.
struct TrainingMetadata
{
    std::uint64_t training_seed      = 0;
    std::uint32_t training_positions = 0;
    std::uint32_t training_passes    = 0;
    std::string   git_sha;
};

template <std::size_t DIM>
class Model
{
    static constexpr std::size_t N = 1ULL << DIM;

public:
    // -----------------------------------------------------------------
    //  Construction
    // -----------------------------------------------------------------

    /// Construct a new (untrained) model from reservoir and readout configs.
    Model(const ReservoirConfig& reservoir_cfg,
          const ReadoutArchConfig& cnn_arch,
          std::size_t depth = 1);

    /// Load a trained model from a binary file.
    /// Throws std::runtime_error on format errors or DIM mismatch.
    static Model Load(const std::string& path, const Vocabulary& vocab);

    // -----------------------------------------------------------------
    //  Reservoir driving
    // -----------------------------------------------------------------

    /// Reset the reservoir to its zero state (no history).
    void Reset();

    /// Advance the reservoir by one character.
    void Step(char ch);

    /// Advance the reservoir through an entire string (no state collection).
    void Warmup(const std::string& text);

    // -----------------------------------------------------------------
    //  Readout — online training
    // -----------------------------------------------------------------

    /// Collect standardization statistics and build the CNN architecture.
    /// Must be called before any training or prediction.
    /// @param text  Text to stream through the reservoir for standardization.
    void InitReadout(const std::string& text, const ReadoutArchConfig& arch);

    /// Copy the current live reservoir state into a caller-owned buffer.
    /// @param out  Must have room for NumOutputVerts() floats.
    void CopyLiveState(float* out) const;

    /// Train the CNN readout on a mini-batch of accumulated states.
    void TrainBatch(const float* states, const int* targets,
                    std::size_t count, float lr);

    // -----------------------------------------------------------------
    //  Readout — prediction
    // -----------------------------------------------------------------

    /// Write raw logits (one per class) from the current live state.
    /// @param logits  Must have room for NumOutputs() floats.
    void Predict(float* logits) const;

    // -----------------------------------------------------------------
    //  State snapshot (for eval during training)
    // -----------------------------------------------------------------

    /// Save the current reservoir state so it can be restored after eval.
    void SaveState(std::vector<float>& state, std::vector<float>& output) const;

    /// Restore a previously saved reservoir state.
    void RestoreState(const std::vector<float>& state,
                      const std::vector<float>& output);

    // -----------------------------------------------------------------
    //  Persistence
    // -----------------------------------------------------------------

    /// Save the trained model to a binary file.
    void Save(const std::string& path,
              const Vocabulary& vocab,
              const TrainingMetadata& meta) const;

    // -----------------------------------------------------------------
    //  Accessors
    // -----------------------------------------------------------------

    [[nodiscard]] std::size_t NumOutputVerts() const { return esn_.NumOutputVerts(); }
    [[nodiscard]] std::size_t NumOutputs()     const { return esn_.NumOutputs(); }
    [[nodiscard]] std::size_t OutputSize()     const { return esn_.OutputSize(); }

    [[nodiscard]] const ReservoirConfig&   ReservoirCfg() const { return reservoir_cfg_; }
    [[nodiscard]] const ReadoutArchConfig& ReadoutCfg()   const { return cnn_arch_; }

private:
    ESN<DIM>          esn_;
    ReservoirConfig   reservoir_cfg_;
    ReadoutArchConfig cnn_arch_;
    std::size_t       depth_;

    float step_bits_[Vocabulary::kInputBits]{};
};

}  // namespace lm
