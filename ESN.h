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

enum class ReadoutType { Linear, Ridge };
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
/// @tparam DIM Hypercube dimension (5-12). Vertex count is 2^DIM.
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

    /// @brief Clear collected states and cached features.
    void ClearStates();

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

    /// @brief Incremental training for streaming (Linear readout only).
    void TrainIncremental(const float* targets, size_t train_size,
                          float blend = 0.1f,
                          float lr = 0.0f, size_t epochs = 200,
                          float weight_decay = 1e-4f, float lr_decay = 0.01f);

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    /// @brief Predict from collected state at a given timestep index.
    [[nodiscard]] float PredictRaw(size_t timestep) const;

    /// @brief R² on a slice of collected states [start, start+count).
    /// Both features and targets are indexed from `start`.
    [[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;

    /// @brief NRMSE on a slice of collected states [start, start+count).
    /// Both features and targets are indexed from `start`.
    [[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;

    /// @brief Classification accuracy on a slice of collected states.
    /// Both features and labels are indexed from `start`.
    [[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;

    // ---------------------------------------------------------------
    //  State & feature access
    // ---------------------------------------------------------------

    /// @brief Extract stride-selected vertices from collected states.
    [[nodiscard]] std::vector<float> SelectedStates() const;

    /// @brief Build and cache features from collected states (incremental).
    /// Only computes features for states added since the last call.
    void EnsureFeatures() const;

    /// @brief Number of features per timestep (M for Raw, 2.5M for Translated).
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

private:
    std::unique_ptr<Reservoir<DIM>> reservoir_;
    ReadoutType readout_type_;
    FeatureMode feature_mode_;
    std::variant<LinearReadout, RidgeRegression> readout_;

    size_t num_inputs_ = 1;
    float output_fraction_ = 1.0f;
    size_t output_stride_ = 1;
    size_t num_output_verts_ = N;

    std::vector<float> states_;      // flat: num_collected_ * N floats
    size_t num_collected_ = 0;

    mutable std::vector<float> features_;    // flat: num_collected_ * NumFeatures() floats
    mutable size_t features_computed_ = 0;   // incremental: features valid for [0, features_computed_)
};
