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
// Mackey-Glass delay differential equation (tau=17)
// ---------------------------------------------------------------------------

/// @brief Generate a Mackey-Glass chaotic time series.
///
/// The Mackey-Glass delay differential equation is a standard benchmark for
/// chaotic time series prediction in reservoir computing:
///
///   dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
///
/// With tau=17, n=10, beta=0.2, gamma=0.1, the system exhibits low-dimensional
/// chaos — deterministic but aperiodic, making it ideal for testing whether a
/// reservoir can track complex nonlinear dynamics.
///
/// The series is integrated with Euler stepping (dt=1) from a flat initial
/// condition (x=1.2). The first 1000+tau transient steps are discarded so the
/// returned values lie on the attractor.
///
/// @param total_steps Number of attractor samples to return.
/// @return Vector of total_steps floats (unnormalized; call Normalize() for [-1,1]).
inline std::vector<float> GenerateMackeyGlass(size_t total_steps)
{
    constexpr size_t TAU = 17;
    constexpr float BETA = 0.2f, GAMMA = 0.1f, N_EXP = 10.0f, DT = 1.0f;

    size_t total_with_history = total_steps + TAU + 1000;
    std::vector<float> x(total_with_history, 1.2f);

    for (size_t t = TAU; t < total_with_history - 1; ++t)
    {
        float x_tau = x[t - TAU];
        float dx = BETA * x_tau / (1.0f + std::pow(x_tau, N_EXP)) - GAMMA * x[t];
        x[t + 1] = x[t] + DT * dx;
    }

    std::vector<float> result(total_steps);
    size_t offset = total_with_history - total_steps;
    for (size_t t = 0; t < total_steps; ++t)
        result[t] = x[offset + t];
    return result;
}

// ---------------------------------------------------------------------------
// Min-max normalize to [-1, 1]
// ---------------------------------------------------------------------------

/// @brief Normalize a signal to the [-1, +1] range (min-max scaling).
///
/// Reservoir inputs are clamped to [-1, +1], so raw signals must be scaled
/// to this range before injection. This function applies min-max normalization:
///   x_normalized = (x - midpoint) / half_range
/// where midpoint = (max+min)/2 and half_range = (max-min)/2.
///
/// If the signal is constant (zero range), it is left unchanged.
inline void Normalize(std::vector<float>& series)
{
    if (series.empty()) return;
    float lo = series[0], hi = series[0];
    for (float v : series) { if (v < lo) lo = v; if (v > hi) hi = v; }
    float half = (hi - lo) / 2.0f;
    if (half < 1e-12f) return;
    float mid = (hi + lo) / 2.0f;
    for (float& v : series) v = (v - mid) / half;
}

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
