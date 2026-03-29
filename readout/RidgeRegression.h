#pragma once

#include <cstddef>
#include <cmath>
#include <vector>

/// @brief Closed-form Ridge regression readout — the mathematically optimal
///        linear readout for reservoir computing evaluation.
///
/// Where LinearReadout iterates toward a solution via SGD, Ridge regression
/// solves for the globally optimal weights in one shot by solving the normal
/// equations with L2 regularization:
///
///   [W; b] = (X'X + lambda * I)^{-1} X'y
///
/// This gives the unique weight vector that minimizes squared error plus a
/// penalty on weight magnitude. The result is deterministic — no learning
/// rate, no epochs, no random initialization.
///
/// **Why Ridge and not plain least squares?** When the number of features
/// approaches or exceeds the number of training samples (common in RC —
/// a DIM=8 reservoir with translation has 640 features), the X'X matrix
/// becomes ill-conditioned or singular. The lambda*I term (default lambda=1.0)
/// adds a small positive value to the diagonal, ensuring the system is
/// always solvable and preventing the weights from exploding to fit noise.
///
/// **How the solve works.** The feature matrix X is augmented with a bias
/// column of 1.0s. X'X is accumulated in double precision for numerical
/// stability, then solved via Gaussian elimination with partial pivoting.
/// The bias weight is NOT regularized — it should be free to shift the
/// output without penalty. X'X computation is OpenMP-parallelized; each
/// thread owns a row of the upper triangle, which is mirrored afterward.
///
/// **Feature standardization.** All features are standardized (zero mean,
/// unit variance) before solving. This ensures the lambda*I penalty treats
/// all features equally regardless of scale — without this, the
/// translation layer's mixed-scale features (x in [-1,1], x² in [0,1],
/// x*x' in [-1,1]) would be regularized unevenly, biasing the solution
/// toward whichever feature group has the largest raw variance. The learned
/// mean and scale are applied automatically during prediction.
///
/// **Cost:** O(N² * samples) to build X'X, O(N³) to solve. This is fast
/// for reservoir-scale problems (N up to ~2500 features) and provides the
/// optimal solution that SGD can only approximate. Preferred for DIM >= 8;
/// for smaller reservoirs, LinearReadout is lighter-weight and sufficient.
///
/// **Interface conventions:**
///   - Features: float, any range (standardized internally)
///   - Targets: float — continuous for regression, {-1, +1} for classification
///   - PredictRaw(): continuous output; Predict(): thresholded at 0

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
