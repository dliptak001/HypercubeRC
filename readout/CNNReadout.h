#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace hcnn { class HCNN; }

/// Configuration for the CNN readout's architecture and training.
struct CNNReadoutConfig {
    int conv_channels = 16;       ///< Number of convolution channels.
    int epochs        = 200;      ///< Training epochs.
    int batch_size    = 32;       ///< Mini-batch size.
    float lr_max      = 0.005f;   ///< Peak learning rate (cosine annealing).
    float lr_min_frac = 0.1f;     ///< lr_min = lr_max * lr_min_frac.
    float weight_decay = 0.0f;    ///< L2 weight decay.
    unsigned seed     = 42;       ///< Weight initialization seed.
};

/// @brief HypercubeCNN-based readout for reservoir computing.
///
/// Replaces LinearReadout / RidgeRegression with a learned convolutional
/// readout that operates directly on raw reservoir state (N = 2^DIM floats
/// per timestep).  The CNN's learned convolution kernels discover which
/// vertex interactions predict the target -- no hand-crafted feature
/// extraction, no translation layer, no stride selection.
///
/// **Data path:** raw reservoir state -> input standardization ->
///   HypercubeCNN (Conv -> MaxPool -> GAP -> Linear) -> de-center -> output.
///
/// **Interface compatibility:** Provides the same method signatures as
/// LinearReadout / RidgeRegression so that ESN's std::visit lambdas compile
/// for prediction, evaluation, and serialization.  Training is handled via
/// a dedicated ESN code path (not through the generic Train visitor) because
/// CNN training requires multi-epoch iteration with its own hyperparameters.
///
/// **PIMPL:** The hcnn::HCNN object is held via unique_ptr behind a forward
/// declaration.  #include "HCNN.h" stays in the .cpp only.
class CNNReadout
{
public:
    CNNReadout();
    ~CNNReadout();
    CNNReadout(CNNReadout&&) noexcept;
    CNNReadout& operator=(CNNReadout&&) noexcept;

    // Non-copyable (owns a unique_ptr<HCNN> which is non-copyable).
    CNNReadout(const CNNReadout&) = delete;
    CNNReadout& operator=(const CNNReadout&) = delete;

    /// @brief Train the CNN readout on raw reservoir states.
    /// @param states     Row-major: num_samples rows, each of N = 2^dim floats.
    /// @param targets    One scalar target per sample.
    /// @param num_samples Number of training samples.
    /// @param dim        Hypercube dimension (N = 2^dim vertices per state).
    /// @param config     Architecture and training hyperparameters.
    void Train(const float* states, const float* targets,
               size_t num_samples, size_t dim,
               const CNNReadoutConfig& config);

    /// @brief Compatibility Train matching Linear/Ridge signature.
    /// Called by ESN's generic std::visit path.  For CNN this is a no-op
    /// that throws -- CNN training must go through the dim-aware overload
    /// or the ESN's CNN-specific Train path.
    void Train(const float* features, const float* labels,
               size_t num_samples, size_t num_features);

    /// @brief Predict from a single raw state vector (N floats).
    /// Applies input standardization and target de-centering.
    [[nodiscard]] float PredictRaw(const float* state) const;

    /// @brief Classification-style thresholded prediction (at 0.0).
    [[nodiscard]] float Predict(const float* state) const;

    /// @brief R-squared on raw state vectors.
    [[nodiscard]] double R2(const float* states, const float* targets,
                            size_t num_samples) const;

    /// @brief Classification accuracy on raw state vectors.
    [[nodiscard]] double Accuracy(const float* states, const float* labels,
                                  size_t num_samples) const;

    // --- State accessors (interface compatibility with Linear/Ridge) ---

    [[nodiscard]] size_t NumFeatures() const { return num_features_; }
    [[nodiscard]] double Bias() const { return target_mean_; }

    [[nodiscard]] const std::vector<float>& FeatureMean() const { return input_mean_; }
    [[nodiscard]] const std::vector<float>& FeatureScale() const { return input_scale_; }

    /// @brief Flattened CNN weights (opaque blob for serialization).
    [[nodiscard]] const std::vector<double>& Weights() const { return weights_blob_; }

    /// @brief Restore a previously trained state.
    void SetState(std::vector<double> weights, double bias,
                  std::vector<float> feature_mean, std::vector<float> feature_scale);

    /// @brief Check if the readout has been trained.
    [[nodiscard]] bool IsTrained() const { return trained_; }

    /// @brief Get the config used for training (valid after Train).
    [[nodiscard]] const CNNReadoutConfig& GetConfig() const { return config_; }

private:
    std::unique_ptr<hcnn::HCNN> net_;
    CNNReadoutConfig config_;
    bool trained_ = false;
    size_t dim_ = 0;
    size_t num_features_ = 0;  // N = 2^dim (raw state size, for interface compat)

    // Input standardization: per-vertex mean and 1/std.
    std::vector<float> input_mean_;
    std::vector<float> input_scale_;

    // Target centering.
    double target_mean_ = 0.0;

    // Flattened weight blob for serialization (mirrors Weights() interface).
    std::vector<double> weights_blob_;

    // Persistent scratch buffers for prediction (zero per-call allocation).
    mutable std::vector<float> scratch_state_;
    mutable std::vector<float> scratch_embedded_;
    mutable std::vector<float> scratch_pred_;

    void standardize(const float* in, float* out, size_t n) const;
    void compute_standardization(const float* states, size_t num_samples, size_t n);
    void flatten_weights();
    void rebuild_from_blob();
};
