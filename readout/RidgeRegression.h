#pragma once

#include <cstddef>
#include <cmath>
#include <vector>

/// @brief Ridge regression readout — optimal linear readout for RC evaluation.
///
/// Finds the globally optimal linear mapping from reservoir states to targets by
/// solving [W; b] = (X'X + λI)^{-1} X'y via Gaussian elimination with partial pivoting.
/// The feature matrix is internally augmented with a bias column of 1.0s; the bias is
/// NOT regularized (only feature weights are penalized), so the intercept is free to
/// shift the decision boundary.
///
/// Public interface accepts float (matching Reservoir and LinearReadout). Matrix math
/// (X'X accumulation, solve, back-substitution) is performed in double precision for
/// numerical stability. X'X computation is OpenMP-parallelized; each thread computes
/// its own row of the upper triangle, which is then mirrored — no write contention.
///
/// Cost: O(N² * samples) for X'X, O(N³) for the solve. Use for DIM >= 8 where
/// the closed-form optimum outperforms SGD. Use LinearReadout for DIM < 8.
///
/// Features are standardized (zero mean, unit variance) before solving. This ensures
/// the regularization term λI penalizes all feature weights equally regardless of
/// scale — critical for the translation layer which produces mixed-scale features.
///
/// All values follow the pipeline convention:
///   - Features: float, any range (standardized internally)
///   - Targets: float, continuous values for regression or {-1.0, +1.0} for classification
///   - PredictRaw: continuous output; Predict: thresholded at 0.0 to {-1.0, +1.0}

class RidgeRegression
{
public:
    /// @brief Train on real-valued features with continuous or {-1, +1} targets.
    /// @param features  Row-major: num_samples rows, each of num_features float values.
    /// @param labels    One label (-1.0f or +1.0f) per sample.
    /// @param num_samples   Number of training samples.
    /// @param num_features  Number of features per sample.
    /// @param lambda        Regularization strength.
    void Train(const float* features, const float* labels,
               size_t num_samples, size_t num_features, double lambda = 1.0);

    /// @brief Classify a single feature vector (threshold at 0.0).
    /// @return +1.0f or -1.0f.
    [[nodiscard]] float Predict(const float* features) const;

    /// @brief Raw prediction value (no threshold).
    [[nodiscard]] float PredictRaw(const float* features) const;

    /// @brief Classify multiple samples and return accuracy.
    [[nodiscard]] double Accuracy(const float* features, const float* labels,
                                  size_t num_samples) const;

    /// @brief R-squared (coefficient of determination) on continuous targets.
    /// Uses PredictRaw (no threshold). Returns 1 - SS_res/SS_tot. Can be negative.
    [[nodiscard]] double R2(const float* features, const float* targets,
                            size_t num_samples) const;

    /// @brief Access the learned weights (double precision, excludes bias).
    [[nodiscard]] const std::vector<double>& Weights() const { return weights_; }

    /// @brief Access the learned bias.
    [[nodiscard]] double Bias() const { return bias_; }

    /// @brief Number of features this model was trained on.
    [[nodiscard]] size_t NumFeatures() const { return num_features_; }

private:
    std::vector<double> weights_;
    double bias_ = 0.0;
    size_t num_features_ = 0;

    // Feature standardization: computed from training set, applied in PredictRaw.
    std::vector<float> feature_mean_;
    std::vector<float> feature_scale_;  // 1/std per feature
};
