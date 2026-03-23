#pragma once

#include <cstddef>
#include <vector>

/// @brief Online linear readout via LMS/SGD — preferred readout for DIM < 8.
///
/// Trains a single-layer linear model (weights + bias) using the least mean squares
/// (Widrow-Hoff) update rule with MSE-based pocket: after each epoch, saves the weight
/// vector if MSE is the lowest seen so far. Returns the pocket (best) weights.
///
/// Features are standardized (zero mean, unit variance) before training. The mean and
/// scale are computed from the training set and applied automatically in PredictRaw.
/// This is critical for the translation layer which produces mixed-scale features:
/// x in [-1,1], x² in [0,1] (biased), x*x' in [-1,1].
///
/// L2 weight decay (default 1e-4) acts as online Ridge regularization. Learning rate
/// decay (default 0.01) ensures convergence. Bias is not decayed.
///
/// All values follow the pipeline convention:
///   - Features: float, any range (standardized internally)
///   - Targets: float, continuous values for regression or {-1.0, +1.0} for classification
///   - PredictRaw: continuous output; Predict: thresholded at 0.0 to {-1.0, +1.0}

class LinearReadout
{
public:
    /// @brief Train on real-valued features with continuous or {-1, +1} targets.
    /// @param learning_rate  Step size. Default 0 = auto (1.0/num_features).
    void Train(const float* features, const float* labels,
               size_t num_samples, size_t num_features,
               float learning_rate = 0.0f, size_t max_epochs = 200,
               float weight_decay = 1e-4f, float lr_decay = 0.01f);

    /// @brief Incrementally update weights from new data (streaming mode).
    ///
    /// Trains a fresh model on the new data, then blends it with the existing model:
    ///   W_updated = (1 - blend) * W_existing + blend * W_new
    ///
    /// Feature standardization stats (mean, scale) are blended the same way, allowing
    /// the model to track distribution drift over time.
    ///
    /// If no prior Train() has been called, delegates to Train() (blend is ignored).
    ///
    /// @param blend  Blending factor in (0, 1]. 1.0 = full replacement (same as Train).
    void TrainIncremental(const float* features, const float* labels,
                          size_t num_samples, size_t num_features,
                          float blend = 0.1f,
                          float learning_rate = 0.0f, size_t max_epochs = 200,
                          float weight_decay = 1e-4f, float lr_decay = 0.01f);

    /// @brief Classify a single feature vector (threshold at 0.0).
    [[nodiscard]] float Predict(const float* features) const;

    /// @brief Raw prediction value (no threshold). Applies feature standardization.
    [[nodiscard]] float PredictRaw(const float* features) const;

    /// @brief Classify multiple samples and return accuracy.
    [[nodiscard]] double Accuracy(const float* features, const float* labels,
                                  size_t num_samples) const;

    /// @brief R-squared (coefficient of determination) on continuous targets.
    [[nodiscard]] double R2(const float* features, const float* targets,
                            size_t num_samples) const;

    /// @brief Access the learned weights (in standardized feature space).
    [[nodiscard]] const std::vector<float>& Weights() const { return weights_; }

    /// @brief Access the learned bias.
    [[nodiscard]] float Bias() const { return bias_; }

    /// @brief Number of features this model was trained on.
    [[nodiscard]] size_t NumFeatures() const { return num_features_; }

private:
    std::vector<float> weights_;
    float bias_ = 0.0f;
    size_t num_features_ = 0;

    // Feature standardization: computed from training set, applied in PredictRaw.
    std::vector<float> feature_mean_;   // per-feature mean
    std::vector<float> feature_scale_;  // per-feature 1/std (or 1.0 if std < epsilon)

    /// @brief Mean squared error over a dataset (using PredictRaw).
    [[nodiscard]] double ComputeMSE(const float* features, const float* targets,
                                     size_t num_samples) const;

    /// @brief Count correct classifications over a dataset.
    [[nodiscard]] size_t CountCorrect(const float* features, const float* labels,
                                      size_t num_samples) const;
};
