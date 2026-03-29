#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstddef>
#include <cmath>
#include <limits>
#include "../ESN.h"
#include "../Reservoir.h"
#include "../TranslationLayer.h"
#include "../SignalGenerators.h"

/// @brief Diagnostic: Mackey-Glass chaotic time series prediction.
///
/// Mackey-Glass delay differential equation (tau=17, n=10, beta=0.2, gamma=0.1)
/// produces low-dimensional chaos. Task: predict x(t+horizon) from reservoir states.
///
/// Reports NRMSE for both raw (N) and full translation (2.5N) features,
/// 3-seed average, LinearReadout.
/// Standard ESN NRMSE on MG h=1: 0.01-0.05 (lower is better).
template <size_t DIM>
class MackeyGlass
{
    static constexpr size_t N = 1ULL << DIM;
    static constexpr size_t FEATURES = TranslationFeatureCount<DIM>();

public:
    MackeyGlass(size_t prediction_horizon = 1)
        : prediction_horizon_(prediction_horizon)
    {
    }

    void RunAndPrint()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        PrintHeader(warmup, collect);

        double s_nrmse_raw = 0.0, s_nrmse_full = 0.0;

        for (uint64_t seed : Seeds())
        {
            auto series = GenerateMackeyGlass(warmup + collect + prediction_horizon_ + 20);
            Normalize(series);

            std::vector<float> targets(collect);
            for (size_t t = 0; t < collect; ++t)
                targets[t] = series[warmup + t + prediction_horizon_];

            size_t tr = static_cast<size_t>(collect * 0.7);
            size_t te = collect - tr;

            // Raw features — raw-optimized defaults
            {
                ESN<DIM> esn(seed, ReadoutType::Linear);
                esn.Warmup(series.data(), warmup);
                esn.Run(series.data() + warmup, collect);
                LinearReadout lr_raw;
                lr_raw.Train(esn.States(), targets.data(), tr, N);
                s_nrmse_raw += ComputeNRMSE(lr_raw, esn.States() + tr * N,
                                              targets.data() + tr, te, N);
            }

            // Full translation — translation-optimized defaults
            {
                ESN<DIM> esn(seed, ReadoutType::Linear, FeatureMode::Translation);
                esn.Warmup(series.data(), warmup);
                esn.Run(series.data() + warmup, collect);
                auto translated = TranslationTransform<DIM>(esn.States(), collect);
                LinearReadout lr_full;
                lr_full.Train(translated.data(), targets.data(), tr, FEATURES);
                s_nrmse_full += ComputeNRMSE(lr_full, translated.data() + tr * FEATURES,
                                              targets.data() + tr, te, FEATURES);
            }
        }

        double n = static_cast<double>(Seeds().size());
        double nrmse_raw = s_nrmse_raw / n;
        double nrmse_full = s_nrmse_full / n;
        double pct = (nrmse_raw > 1e-12) ? 100.0 * (nrmse_full - nrmse_raw) / nrmse_raw : 0.0;

        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(5) << N
                  << " | " << std::fixed << std::setprecision(4) << std::setw(7) << nrmse_raw
                  << " | " << std::setprecision(4) << std::setw(7) << nrmse_full
                  << " (" << std::showpos << std::setprecision(1) << std::setw(5) << pct
                  << "%" << std::noshowpos << ")\n";
    }

private:
    size_t prediction_horizon_;

    static std::vector<uint64_t> Seeds() { return {42, 1042, 2042}; }

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
        if (var < 1e-12) return std::numeric_limits<double>::infinity();
        return std::sqrt(mse / num_samples) / std::sqrt(var / num_samples);
    }

    void PrintHeader(size_t warmup, size_t collect) const
    {
        std::cout << "=== Mackey-Glass h=" << prediction_horizon_
                  << " (LinearReadout, 3-seed avg, raw vs full translation) ===\n";
        std::cout << "Seeds: {42,1042,2042} | Alpha: 1.0 | Leak: 1.0"
                  << " | SR: per-DIM default | Input scaling: per-DIM default\n";
        std::cout << "Warmup: " << warmup << " | Collect: " << collect
                  << " | Features: " << N << " raw, " << FEATURES << " translated\n\n";

        std::cout << "  DIM |     N |    raw  |   full\n";
        std::cout << "  ----+-------+---------+-----------------\n";
    }
};
