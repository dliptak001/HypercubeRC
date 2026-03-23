#include "RidgeRegression.h"
#include <algorithm>
#include <omp.h>

void RidgeRegression::Train(const float* features, const float* labels,
                             size_t num_samples, size_t num_features, double lambda)
{
    num_features_ = num_features;
    weights_.assign(num_features, 0.0);
    bias_ = 0.0;

    // Compute per-feature mean and std
    feature_mean_.assign(num_features, 0.0f);
    feature_scale_.assign(num_features, 1.0f);

    for (size_t s = 0; s < num_samples; ++s)
    {
        const float* x = features + s * num_features;
        for (size_t f = 0; f < num_features; ++f)
            feature_mean_[f] += x[f];
    }
    for (size_t f = 0; f < num_features; ++f)
        feature_mean_[f] /= static_cast<float>(num_samples);

    std::vector<float> var(num_features, 0.0f);
    for (size_t s = 0; s < num_samples; ++s)
    {
        const float* x = features + s * num_features;
        for (size_t f = 0; f < num_features; ++f)
        {
            float d = x[f] - feature_mean_[f];
            var[f] += d * d;
        }
    }
    for (size_t f = 0; f < num_features; ++f)
    {
        float std_f = std::sqrt(var[f] / static_cast<float>(num_samples));
        feature_scale_[f] = (std_f > 1e-8f) ? 1.0f / std_f : 1.0f;
    }

    // Standardize features into working buffer
    std::vector<float> std_features(num_samples * num_features);
    for (size_t s = 0; s < num_samples; ++s)
    {
        const float* x = features + s * num_features;
        float* xn = std_features.data() + s * num_features;
        for (size_t f = 0; f < num_features; ++f)
            xn[f] = (x[f] - feature_mean_[f]) * feature_scale_[f];
    }

    const float* sf = std_features.data();

    // Augmented feature count: original features + 1 bias column
    const size_t aug = num_features + 1;

    // Compute X'X + lambda*I on standardized features
    std::vector<double> XtX(aug * aug, 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_features; ++i)
        for (size_t k = i; k < num_features; ++k)
        {
            double sum = 0.0;
            for (size_t s = 0; s < num_samples; ++s)
                sum += static_cast<double>(sf[s * num_features + i])
                     * static_cast<double>(sf[s * num_features + k]);
            XtX[i * aug + k] = sum;
        }

    for (size_t i = 0; i < num_features; ++i)
        for (size_t k = i + 1; k < num_features; ++k)
            XtX[k * aug + i] = XtX[i * aug + k];

    // Feature-bias cross terms (standardized features have ~zero mean, but compute exactly)
    for (size_t i = 0; i < num_features; ++i)
    {
        double sum = 0.0;
        for (size_t s = 0; s < num_samples; ++s)
            sum += static_cast<double>(sf[s * num_features + i]);
        XtX[i * aug + num_features] = sum;
        XtX[num_features * aug + i] = sum;
    }

    XtX[num_features * aug + num_features] = static_cast<double>(num_samples);

    for (size_t i = 0; i < num_features; ++i)
        XtX[i * aug + i] += lambda;

    // Compute X'y on standardized features
    std::vector<double> Xty(aug, 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_features; ++i)
        for (size_t s = 0; s < num_samples; ++s)
            Xty[i] += static_cast<double>(sf[s * num_features + i])
                     * static_cast<double>(labels[s]);

    for (size_t s = 0; s < num_samples; ++s)
        Xty[num_features] += static_cast<double>(labels[s]);

    // Solve (X'X + lambda*I) w = X'y via Gaussian elimination with partial pivoting
    size_t cols = aug + 1;
    std::vector<double> A(aug * cols);
    for (size_t i = 0; i < aug; ++i)
    {
        for (size_t k = 0; k < aug; ++k)
            A[i * cols + k] = XtX[i * aug + k];
        A[i * cols + aug] = Xty[i];
    }

    for (size_t col = 0; col < aug; ++col)
    {
        // Partial pivoting
        size_t max_row = col;
        double max_val = std::abs(A[col * cols + col]);
        for (size_t row = col + 1; row < aug; ++row)
        {
            double v = std::abs(A[row * cols + col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_row != col)
            for (size_t k = col; k <= aug; ++k)
                std::swap(A[col * cols + k], A[max_row * cols + k]);

        double pivot = A[col * cols + col];
        if (std::abs(pivot) < 1e-12) continue;

        // Forward elimination
        #pragma omp parallel for schedule(static)
        for (size_t row = col + 1; row < aug; ++row)
        {
            double factor = A[row * cols + col] / pivot;
            for (size_t k = col; k <= aug; ++k)
                A[row * cols + k] -= factor * A[col * cols + k];
        }
    }

    // Back substitution into a temporary augmented weight vector
    std::vector<double> w_aug(aug, 0.0);
    for (int i = static_cast<int>(aug) - 1; i >= 0; --i)
    {
        double sum = A[i * cols + aug];
        for (size_t k = i + 1; k < aug; ++k)
            sum -= A[i * cols + k] * w_aug[k];
        double diag = A[i * cols + i];
        w_aug[i] = (std::abs(diag) > 1e-12) ? sum / diag : 0.0;
    }

    // Split augmented weights into feature weights + bias
    for (size_t i = 0; i < num_features; ++i)
        weights_[i] = w_aug[i];
    bias_ = w_aug[num_features];
}

float RidgeRegression::Predict(const float* features) const
{
    return PredictRaw(features) >= 0.0f ? 1.0f : -1.0f;
}

float RidgeRegression::PredictRaw(const float* features) const
{
    double sum = bias_;
    for (size_t i = 0; i < num_features_; ++i)
        sum += weights_[i] * static_cast<double>((features[i] - feature_mean_[i]) * feature_scale_[i]);
    return static_cast<float>(sum);
}

double RidgeRegression::Accuracy(const float* features, const float* labels,
                                  size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    size_t correct = 0;
    for (size_t s = 0; s < num_samples; ++s)
        if (Predict(features + s * num_features_) == labels[s])
            ++correct;
    return static_cast<double>(correct) / num_samples;
}

double RidgeRegression::R2(const float* features, const float* targets,
                            size_t num_samples) const
{
    if (num_samples == 0) return 0.0;

    double mean = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
        mean += targets[s];
    mean /= num_samples;

    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
    {
        double y = targets[s];
        double y_hat = PredictRaw(features + s * num_features_);
        ss_tot += (y - mean) * (y - mean);
        ss_res += (y - y_hat) * (y - y_hat);
    }

    if (ss_tot < 1e-12) return 0.0;
    return 1.0 - ss_res / ss_tot;
}
