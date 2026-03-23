#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstddef>
#include <cmath>
#include "../ESN.h"
#include "../TranslationLayer.h"

/// @brief Diagnostic: NARMA-10 nonlinear benchmark.
///
/// NARMA-10 (Nonlinear AutoRegressive Moving Average, order 10) is the standard
/// RC benchmark for combined memory + nonlinear computation. The target is:
///
///   y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1
///
/// Reports NRMSE for both raw (N) and full translation (2.5N) features,
/// 3-seed average, LinearReadout.
/// Standard ESN NRMSE on NARMA-10: 0.2-0.4 (lower is better).
template <size_t DIM>
class NARMA10
{
    static constexpr size_t N = 1ULL << DIM;
    static constexpr size_t FEATURES = TranslationFeatureCount<DIM>();

public:
    NARMA10() = default;

    void RunAndPrint()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        PrintHeader(warmup, collect);

        double s_nrmse_raw = 0.0, s_nrmse_full = 0.0;

        for (uint64_t seed : Seeds())
        {
            auto [u, y] = GenerateNARMA10(seed + 99, warmup + collect);

            // Scale inputs to [-1, 1] for the reservoir (NARMA uses [0, 0.5])
            std::vector<float> ri(warmup + collect);
            for (size_t t = 0; t < ri.size(); ++t)
                ri[t] = u[t] * 4.0f - 1.0f;

            ESN<DIM> esn(seed, ReadoutType::Linear);
            esn.Warmup(ri.data(), warmup);
            esn.Run(ri.data() + warmup, collect);

            const float* states = esn.States();

            std::vector<float> targets(collect);
            for (size_t t = 0; t < collect; ++t)
                targets[t] = y[warmup + t];

            size_t tr = static_cast<size_t>(collect * 0.7);
            size_t te = collect - tr;

            // Raw features
            LinearReadout lr_raw;
            lr_raw.Train(states, targets.data(), tr, N);
            s_nrmse_raw += ComputeNRMSE(lr_raw, states + tr * N,
                                          targets.data() + tr, te, N);

            // Full translation
            auto translated = TranslationTransform<DIM>(states, collect);
            LinearReadout lr_full;
            lr_full.Train(translated.data(), targets.data(), tr, FEATURES);
            s_nrmse_full += ComputeNRMSE(lr_full, translated.data() + tr * FEATURES,
                                          targets.data() + tr, te, FEATURES);
        }

        double n = static_cast<double>(Seeds().size());
        double nrmse_raw = s_nrmse_raw / n;
        double nrmse_full = s_nrmse_full / n;
        double pct = (nrmse_raw > 1e-12) ? 100.0 * (nrmse_full - nrmse_raw) / nrmse_raw : 0.0;

        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(5) << N
                  << " | " << std::fixed << std::setprecision(3) << std::setw(7) << nrmse_raw
                  << " | " << std::setprecision(3) << std::setw(7) << nrmse_full
                  << " (" << std::showpos << std::setprecision(1) << std::setw(5) << pct
                  << "%" << std::noshowpos << ")\n";
    }

private:
    static std::vector<uint64_t> Seeds() { return {42, 1042, 2042}; }

    struct Sequences
    {
        std::vector<float> inputs;
        std::vector<float> targets;
    };

    static Sequences GenerateNARMA10(uint64_t input_seed, size_t total_steps)
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
            for (size_t i = 0; i < 10; ++i)
                y_sum += y[t - i];

            y[t + 1] = 0.3f * y[t]
                      + 0.05f * y[t] * y_sum
                      + 1.5f * u[t - 9] * u[t]
                      + 0.1f;

            if (y[t + 1] > 1.0f) y[t + 1] = 1.0f;
            if (y[t + 1] < 0.0f) y[t + 1] = 0.0f;
        }

        return {u, y};
    }

    static double ComputeNRMSE(const LinearReadout& readout, const float* features,
                                const float* targets, size_t num_samples, size_t num_features)
    {
        double mean = 0.0;
        for (size_t s = 0; s < num_samples; ++s)
            mean += targets[s];
        mean /= num_samples;

        double var = 0.0, mse = 0.0;
        for (size_t s = 0; s < num_samples; ++s)
        {
            double y = targets[s];
            double y_hat = readout.PredictRaw(features + s * num_features);
            var += (y - mean) * (y - mean);
            mse += (y - y_hat) * (y - y_hat);
        }
        if (var < 1e-12) return 0.0;
        return std::sqrt(mse / num_samples) / std::sqrt(var / num_samples);
    }

    static void PrintHeader(size_t warmup, size_t collect)
    {
        std::cout << "=== NARMA-10 (LinearReadout, 3-seed avg, raw vs full translation) ===\n";
        std::cout << "Seeds: {42,1042,2042} | Alpha: 1.0 | Leak: 1.0"
                  << " | SR: per-DIM default | Input scaling: per-DIM default\n";
        std::cout << "Warmup: " << warmup << " | Collect: " << collect
                  << " | Features: " << N << " raw, " << FEATURES << " translated\n\n";

        std::cout << "  DIM |     N |    raw  |   full\n";
        std::cout << "  ----+-------+---------+-----------------\n";
    }
};
