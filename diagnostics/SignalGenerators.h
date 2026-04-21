#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

/// @brief Shared signal generators for benchmarks and diagnostics.
///
/// Canonical implementations — all benchmarks and diagnostics should use these
/// rather than maintaining local copies.

// ---------------------------------------------------------------------------
// NARMA-10 nonlinear autoregressive benchmark
// ---------------------------------------------------------------------------

/// Input-target pair returned by GenerateNARMA10.
struct NARMASeq { std::vector<float> inputs; std::vector<float> targets; };

/// @brief Generate a NARMA-10 input/target sequence.
///
/// NARMA-10 (Nonlinear AutoRegressive Moving Average, order 10) is the
/// standard reservoir computing benchmark for combined memory and nonlinear
/// computation. The target recurrence is:
///
///   y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1
///
/// The product term u(t-9)*u(t) requires remembering input from 10 steps ago,
/// while the y(t)*sum(...) term requires nonlinear mixing — a reservoir must
/// have both memory depth and computational capacity to perform well.
///
/// Inputs u(t) are uniform random in [0, 0.5]. Targets y(t) are clamped to [0, 1].
///
/// @param input_seed  RNG seed for deterministic input generation.
/// @param total_steps Length of the returned sequences.
/// @return NARMASeq with .inputs (u) and .targets (y), each of total_steps floats.
/// @pre total_steps >= 11 (the recurrence needs 10 history steps).
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

// ---------------------------------------------------------------------------
// NRMSE computation
// ---------------------------------------------------------------------------

/// @brief Normalized Root Mean Squared Error from raw prediction arrays.
///
/// NRMSE = RMSE / std(targets). A value of 1.0 means the model predicts
/// no better than the target mean; 0.0 is perfect. Standard RC benchmarks
/// report NRMSE so results are comparable across different target scales.
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

/// @brief NRMSE from a trained readout (calls PredictRaw per sample).
/// Works with RidgeRegression, CNNReadout, or any type with PredictRaw().
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
