#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

/// Shared signal generators for benchmarks and diagnostics.

struct NARMASeq { std::vector<float> inputs; std::vector<float> targets; };

/// Generate NARMA-10 input/target sequence.
/// y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i),i=0..9) + 1.5*u(t-9)*u(t) + 0.1
/// Inputs u(t) uniform in [0, 0.5]. Targets clamped to [0, 1].
inline NARMASeq GenerateNARMA10(uint64_t input_seed, size_t total_steps)
{
    std::mt19937_64 rng(input_seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> u(total_steps);
    std::vector<float> y(total_steps, 0.0f);

    for (size_t t = 0; t < total_steps; ++t)
        u[t] = static_cast<float>(dist(rng)) * 0.25f + 0.25f;

    for (size_t t = 10; t < total_steps - 1; ++t)
    {
        float y_sum = 0.0f;
        for (size_t i = 0; i < 10; ++i) y_sum += y[t - i];
        y[t + 1] = 0.3f * y[t] + 0.05f * y[t] * y_sum
                  + 1.5f * u[t - 9] * u[t] + 0.1f;
        if (y[t + 1] > 1.0f) y[t + 1] = 1.0f;
        if (y[t + 1] < 0.0f) y[t + 1] = 0.0f;
    }
    return {u, y};
}

/// NRMSE = RMSE / std(targets). 0.0 = perfect, 1.0 = no better than mean.
inline double ComputeNRMSE(const float* pred, const float* targets, size_t n)
{
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += targets[i];
    mean /= static_cast<double>(n);

    double var = 0.0, mse = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double y = targets[i], yh = pred[i];
        var += (y - mean) * (y - mean);
        mse += (y - yh) * (y - yh);
    }
    if (var < 1e-12) return std::numeric_limits<double>::infinity();
    return std::sqrt(mse / n) / std::sqrt(var / n);
}

/// NRMSE from a trained readout (calls PredictRaw per sample).
template <typename Readout>
double ComputeNRMSE(const Readout& readout, const float* features,
                    const float* targets, size_t num_samples, size_t num_features)
{
    double mean = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
        mean += targets[s];
    mean /= static_cast<double>(num_samples);

    double var = 0.0, mse = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
    {
        double y = targets[s];
        double y_hat = readout.PredictRaw(features + s * num_features);
        var += (y - mean) * (y - mean);
        mse += (y - y_hat) * (y - y_hat);
    }
    if (var < 1e-12) return std::numeric_limits<double>::infinity();
    return std::sqrt(mse / num_samples) / std::sqrt(var / num_samples);
}
