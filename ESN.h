#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <variant>
#include <vector>
#include "Reservoir.h"
#include "TranslationLayer.h"
#include "LinearReadout.h"
#include "RidgeRegression.h"
#include "CNNReadout.h"

enum class ReadoutType { Linear, Ridge, HCNN };
enum class FeatureMode { Raw, Translated };

/// @brief Echo-state network implementing the full pipeline:
///        Reservoir -> [Output Selection] -> [Translation] -> Readout.
///
/// ESN owns all stages of the pipeline. Construct with a ReservoirConfig,
/// a ReadoutType, and a FeatureMode, then drive, train, and predict:
///
///     ESN<6> esn(cfg, ReadoutType::Ridge, FeatureMode::Translated);
///     esn.Warmup(inputs, 200);
///     esn.Run(inputs + 200, total);
///     esn.Train(targets, train_size);
///     double r2 = esn.R2(targets, train_size, test_size);
///
/// **Feature modes.**
///   - Raw: M stride-selected vertex states (M features per timestep).
///   - Translated: M selected -> 2.5M features via [x | x² | x*x_antipodal].
///
/// **Training.** Train() uses sensible defaults for both readout types.
/// Power users can call the Ridge overload with a custom lambda, or the
/// Linear overload with custom SGD parameters (lr, epochs).
///
/// **State access.** States(), SelectedStates(), and Features() remain
/// available for direct access (diagnostics, analysis, custom readouts).
///
/// @tparam DIM Hypercube dimension (5-16). Vertex count is 2^DIM.
template <size_t DIM>
class ESN
{
    static constexpr size_t N = 1ULL << DIM;

public:
    explicit ESN(const ReservoirConfig& cfg,
                 ReadoutType readout_type = ReadoutType::Ridge,
                 FeatureMode feature_mode = FeatureMode::Translated);

    // ---------------------------------------------------------------
    //  Reservoir driving
    // ---------------------------------------------------------------

    /// @brief Drive the reservoir without recording.
    /// @param inputs  Pointer to num_steps * num_inputs floats, row-major
    ///                (num_inputs values per timestep).  When num_inputs == 1,
    ///                this is simply num_steps scalars — fully backwards compatible.
    void Warmup(const float* inputs, size_t num_steps);

    /// @brief Drive the reservoir and collect state snapshots.
    /// @param inputs  Pointer to num_steps * num_inputs floats, row-major.
    void Run(const float* inputs, size_t num_steps);

    /// @brief State management — three reset scopes.
    ///
    /// | Method               | Reservoir | Cache |
    /// |----------------------|-----------|-------|
    /// | `ClearStates()`      |   keep    | clear |
    /// | `ResetReservoirOnly()`|  clear   | keep  |
    /// | `Reset()`            |   clear   | clear |
    ///
    /// "Reservoir" here means the live reservoir state (`vtx_state_` /
    /// `vtx_output_`); recurrent and input weights are always preserved.
    /// "Cache" means `states_` + `features_` (the training buffer).
    ///
    /// Pick the one that matches the episode boundary:
    ///   - `ClearStates()`: drop training samples, keep dynamics going
    ///     (e.g. sliding-window retraining on continuous data).
    ///   - `ResetReservoirOnly()`: episodic reservoir, cumulative buffer
    ///     (e.g. HRCCNN_LLM_Math per-expression reset + priming).
    ///   - `Reset()`: completely fresh start (e.g. new validation run).

    /// @brief Clear collected states and cached features; reservoir state untouched.
    void ClearStates();

    /// @brief Zero reservoir state AND clear collected states/features.
    /// Equivalent to `ResetReservoirOnly() + ClearStates()`.
    void Reset();

    /// @brief Zero only the reservoir state; cache (states_, features_) preserved.
    void ResetReservoirOnly();

    // ---------------------------------------------------------------
    //  Training
    // ---------------------------------------------------------------

    /// @brief Train on the first train_size collected states with default parameters.
    void Train(const float* targets, size_t train_size);

    /// @brief Train Ridge readout with custom lambda.
    void Train(const float* targets, size_t train_size, double lambda);

    /// @brief Train Linear readout with custom SGD parameters.
    void Train(const float* targets, size_t train_size,
               float lr, size_t epochs,
               float weight_decay = 1e-4f, float lr_decay = 0.01f);

    /// @brief Train CNN readout on raw reservoir states (bypasses feature pipeline).
    void Train(const float* targets, size_t train_size,
               const CNNReadoutConfig& config);

    /// @brief Train CNN readout with runtime hooks (mid-training eval callback).
    /// See CNNTrainHooks in CNNReadout.h for semantics.
    void Train(const float* targets, size_t train_size,
               const CNNReadoutConfig& config,
               const CNNTrainHooks& hooks);

    /// @brief Incremental training for streaming (Linear readout only).
    void TrainIncremental(const float* targets, size_t train_size,
                          float blend = 0.1f,
                          float lr = 0.0f, size_t epochs = 200,
                          float weight_decay = 1e-4f, float lr_decay = 0.01f);

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    /// @brief Predict from collected state at a given timestep index (scalar).
    [[nodiscard]] float PredictRaw(size_t timestep) const;

    /// @brief Multi-output predict: writes NumOutputs() floats to output.
    /// For HCNN readout with num_outputs > 1; for Linear/Ridge writes 1 float.
    void PredictRaw(size_t timestep, float* output) const;

    /// @brief Predict from the reservoir's CURRENT output, bypassing the
    /// cached states_ buffer. Intended for autoregressive / streaming
    /// inference loops that step the reservoir and immediately predict
    /// without snapshotting state. Scalar overload (num_outputs must be 1).
    [[nodiscard]] float PredictLiveRaw() const;

    /// @brief Multi-output live predict: writes NumOutputs() floats to output.
    /// See scalar overload for semantics.
    void PredictLiveRaw(float* output) const;

    /// @brief R² on a slice of collected states [start, start+count).
    /// targets layout: count * NumOutputs() floats (row-major) for HCNN,
    /// count floats for Linear/Ridge.  Indexed from `start`.
    [[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;

    /// @brief NRMSE on a slice of collected states [start, start+count).
    /// targets layout: count * NumOutputs() floats (row-major) for HCNN,
    /// count floats for Linear/Ridge.  Indexed from `start`.
    [[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;

    /// @brief Classification accuracy on a slice of collected states.
    /// labels layout: count floats (class indices for HCNN, ±1 for Linear/Ridge).
    /// Indexed from `start`.
    [[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;

    /// @brief Number of output neurons (1 for Linear/Ridge, config-based for HCNN).
    [[nodiscard]] size_t NumOutputs() const;

    // ---------------------------------------------------------------
    //  State & feature access
    // ---------------------------------------------------------------

    /// @brief Extract stride-selected vertices from collected states.
    [[nodiscard]] std::vector<float> SelectedStates() const;

    /// @brief Build and cache features from collected states (incremental).
    /// Only computes features for states added since the last call.
    void EnsureFeatures() const;

    /// @brief Size of the per-timestep vector the readout consumes.
    ///   Linear/Ridge + Raw        : M = num_output_verts_
    ///   Linear/Ridge + Translated : 2.5 * M
    ///   HCNN                      : M (CNN operates on the subsampled sub-hypercube)
    [[nodiscard]] size_t NumFeatures() const;

    // --- Accessors ---
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] const float* States() const { return states_.data(); }
    [[nodiscard]] const float* Features() const { return features_.data(); }
    [[nodiscard]] float OutputFraction() const { return output_fraction_; }
    [[nodiscard]] size_t OutputStride() const { return output_stride_; }
    [[nodiscard]] size_t NumOutputVerts() const { return num_output_verts_; }
    [[nodiscard]] ReadoutType GetReadoutType() const { return readout_type_; }
    [[nodiscard]] FeatureMode GetFeatureMode() const { return feature_mode_; }
    [[nodiscard]] float GetAlpha() const { return reservoir_->GetAlpha(); }
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

    // --- Config & persistence ---

    /// @brief Reconstruct the full ReservoirConfig used to create this ESN.
    [[nodiscard]] ReservoirConfig GetConfig() const;

    /// @brief Trained readout state — everything needed to serialize/restore predictions.
    struct ReadoutState {
        std::vector<double> weights;     ///< Weight vector (double for both readout types).
        double bias = 0.0;               ///< Bias term.
        std::vector<float> feature_mean; ///< Per-feature mean from training standardization.
        std::vector<float> feature_scale;///< Per-feature 1/std from training standardization.
        bool is_trained = false;         ///< True if the readout has been trained.
    };

    /// @brief Extract the trained readout state for serialization.
    [[nodiscard]] ReadoutState GetReadoutState() const;

    /// @brief Restore a previously trained readout state (from deserialization).
    void SetReadoutState(const ReadoutState& state);

private:
    std::unique_ptr<Reservoir<DIM>> reservoir_;
    ReadoutType readout_type_;
    FeatureMode feature_mode_;
    std::variant<LinearReadout, RidgeRegression, CNNReadout> readout_;

    size_t num_inputs_ = 1;
    float output_fraction_ = 1.0f;
    size_t output_stride_ = 1;
    size_t num_output_verts_ = N;

    std::vector<float> states_;      // flat: num_collected_ * N floats
    size_t num_collected_ = 0;

    mutable std::vector<float> features_;    // flat: num_collected_ * NumFeatures() floats
    mutable size_t features_computed_ = 0;   // incremental: features valid for [0, features_computed_)

    // Sub-hypercube subsampling helpers. Reservoir state is N = 2^DIM floats;
    // consumers (HCNN's CNN and Raw-mode Linear/Ridge live inference) see
    // num_output_verts_ = 2^EffectiveDIM() floats by taking every
    // output_stride_'th vertex. output_stride_ is validated power-of-2 at
    // construction when readout_type_ == HCNN.
    [[nodiscard]] size_t EffectiveDIM() const;
    const float* SubsampleIntoScratch(const float* src) const; // fills scratch_subsampled_
    const float* HCNNState(size_t timestep) const;             // delegates to SubsampleIntoScratch
    [[nodiscard]] std::vector<float> HCNNStates(size_t start, size_t count) const;

    // Per-prediction scratch for one-at-a-time subsampling.
    // NOT thread-safe: concurrent const calls (e.g. parallel PredictRaw over
    // timesteps) would race on this buffer. Use per-thread ESN instances
    // when parallelizing prediction.
    mutable std::vector<float> scratch_subsampled_;
};
