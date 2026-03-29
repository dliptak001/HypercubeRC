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
struct NARMASeq { std::vector<float> inputs; std::vector<float> targets; };

/// @pre total_steps >= 11 (the NARMA recurrence needs 10 history steps).
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
