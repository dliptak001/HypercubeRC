#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstddef>
#include <cmath>
#include "../ESN.h"
#include "../ReservoirDefaults.h"
#include "../TranslationLayer.h"
#include "../SignalGenerators.h"
#include "../readout/LinearReadout.h"
#include "../readout/RidgeRegression.h"

/// @brief Diagnostic: Mackey-Glass chaotic time series prediction.
///
/// Mackey-Glass delay differential equation (tau=17, n=10, beta=0.2, gamma=0.1)
/// produces low-dimensional chaos. Task: predict x(t+horizon) from reservoir states.
///
/// Reports NRMSE for both raw (N) and full translation (2.5N) features,
/// 3-seed average. Readout type (Linear or Ridge) is configurable.
/// Standard ESN NRMSE on MG h=1: 0.01-0.05 (lower is better).
template <size_t DIM>
class MackeyGlass
{
    static constexpr size_t N = 1ULL << DIM;
    static constexpr size_t FEATURES = TranslationFeatureCount<DIM>();

public:
    struct Result
    {
        double nrmse_raw;
        double nrmse_full;
        double pct_change;  // % change from raw to full
    };

    MackeyGlass(size_t prediction_horizon = 1, ReadoutType readout_type = ReadoutType::Linear,
                const ReservoirConfig* config = nullptr)
        : prediction_horizon_(prediction_horizon), readout_type_(readout_type),
          config_(config)
    {
    }

    /// @brief Run the benchmark and return results without printing.
    Result Run()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        double s_nrmse_raw = 0.0, s_nrmse_full = 0.0;

        // Generic: train readout, compute NRMSE — works for LinearReadout or RidgeRegression.
        auto eval = [](auto& readout, const float* features, const float* targets,
                       size_t tr, size_t te, size_t nf)
        {
            readout.Train(features, targets, tr, nf);
            return ComputeNRMSE(readout, features + tr * nf, targets + tr, te, nf);
        };

        for (uint64_t seed : Seeds())
        {
            auto series = GenerateMackeyGlass(warmup + collect + prediction_horizon_ + 20);
            Normalize(series);

            std::vector<float> targets(collect);
            for (size_t t = 0; t < collect; ++t)
                targets[t] = series[warmup + t + prediction_horizon_];

            size_t tr = static_cast<size_t>(collect * 0.7);
            size_t te = collect - tr;

            // Raw features
            {
                auto cfg = config_ ? *config_ : ReservoirDefaults<DIM>::MakeConfig(seed);
                cfg.seed = seed;
                ESN<DIM> esn(cfg, readout_type_);
                esn.Warmup(series.data(), warmup);
                esn.Run(series.data() + warmup, collect);
                if (readout_type_ == ReadoutType::Ridge)
                { RidgeRegression r; s_nrmse_raw += eval(r, esn.States(), targets.data(), tr, te, N); }
                else
                { LinearReadout r; s_nrmse_raw += eval(r, esn.States(), targets.data(), tr, te, N); }
            }

            // Full translation
            {
                auto cfg = config_ ? *config_ : ReservoirDefaults<DIM>::MakeConfig(seed, FeatureMode::Translation);
                cfg.seed = seed;
                ESN<DIM> esn(cfg, readout_type_);
                esn.Warmup(series.data(), warmup);
                esn.Run(series.data() + warmup, collect);
                auto translated = TranslationTransform<DIM>(esn.States(), collect);
                if (readout_type_ == ReadoutType::Ridge)
                { RidgeRegression r; s_nrmse_full += eval(r, translated.data(), targets.data(), tr, te, FEATURES); }
                else
                { LinearReadout r; s_nrmse_full += eval(r, translated.data(), targets.data(), tr, te, FEATURES); }
            }
        }

        double n = static_cast<double>(Seeds().size());
        double nrmse_raw = s_nrmse_raw / n;
        double nrmse_full = s_nrmse_full / n;
        double pct = (nrmse_raw > 1e-12) ? 100.0 * (nrmse_full - nrmse_raw) / nrmse_raw : 0.0;

        return {nrmse_raw, nrmse_full, pct};
    }

    /// @brief Run the benchmark and print a standalone result row.
    void RunAndPrint()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;
        PrintHeader(warmup, collect);

        auto r = Run();

        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(5) << N
                  << " | " << std::fixed << std::setprecision(4) << std::setw(7) << r.nrmse_raw
                  << " | " << std::setprecision(4) << std::setw(7) << r.nrmse_full
                  << " (" << std::showpos << std::setprecision(1) << std::setw(5) << r.pct_change
                  << "%" << std::noshowpos << ")\n";
    }

private:
    size_t prediction_horizon_;
    ReadoutType readout_type_;
    const ReservoirConfig* config_;

    static std::vector<uint64_t> Seeds() { return {42, 1042, 2042}; }

    void PrintHeader(size_t warmup, size_t collect) const
    {
        const char* rn = (readout_type_ == ReadoutType::Ridge) ? "Ridge" : "Linear";
        std::cout << "=== Mackey-Glass h=" << prediction_horizon_
                  << " (" << rn << " Readout, 3-seed avg, raw vs full translation) ===\n";
        std::cout << "Seeds: {42,1042,2042} | Alpha: 1.0 | Leak: 1.0"
                  << " | SR: per-DIM default | Input scaling: per-DIM default\n";
        std::cout << "Warmup: " << warmup << " | Collect: " << collect
                  << " | Features: " << N << " raw, " << FEATURES << " translated\n\n";

        std::cout << "  DIM |     N |    raw  |   full\n";
        std::cout << "  ----+-------+---------+-----------------\n";
    }
};
