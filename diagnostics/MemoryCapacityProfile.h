#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstddef>
#include "../ESN.h"
#include "../ReservoirDefaults.h"
#include "../TranslationLayer.h"
#include "../readout/LinearReadout.h"
#include "../readout/RidgeRegression.h"

/// @brief Diagnostic: Memory capacity profile across lags 1-50.
///
/// For each lag L, fits a LinearReadout from translated reservoir states (2.5N features)
/// to the continuous input value from L steps ago: target[t] = input[t - L].
/// Reports per-lag R2 at selected display lags plus total MC (sum of R2 over all
/// lags 1-50). 3-seed average.
///
/// NOTE: This diagnostic uses full translation (2.5N features), which yields higher MC
/// than the standard metric (raw N states). The main.cpp cascade benchmark reports both
/// raw and full translation side by side.
///
/// This is the standard ESN memory capacity metric from Jaeger (2001), extended
/// with the full translation layer.
template <size_t DIM>
class MemoryCapacityProfile
{
    static constexpr size_t N = 1ULL << DIM;
    static constexpr size_t FEATURES = TranslationFeatureCount<DIM>();
    static constexpr size_t MAX_LAG = 50;

public:
    MemoryCapacityProfile(ReadoutType readout_type = ReadoutType::Linear,
                          const ReservoirConfig* config = nullptr)
        : readout_type_(readout_type), config_(config)
    {
    }

    void RunAndPrint(const std::vector<size_t>& display_lags = {1, 2, 4, 8, 16, 32, 48})
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        PrintHeader(warmup, collect);

        // Accumulate per-lag R2 across seeds
        std::vector<double> r2_sum(MAX_LAG + 1, 0.0);

        for (uint64_t seed : Seeds())
        {
            std::mt19937_64 rng(seed + 99);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            size_t total = warmup + collect;
            std::vector<float> inputs(total);
            for (size_t i = 0; i < total; ++i)
                inputs[i] = static_cast<float>(dist(rng));

            auto cfg = config_ ? *config_ : ReservoirDefaults<DIM>::MakeConfig(seed, FeatureMode::Translation);
            cfg.seed = seed;
            ESN<DIM> esn(cfg, readout_type_);
            esn.Warmup(inputs.data(), warmup);
            esn.Run(inputs.data() + warmup, collect);

            // Apply full translation layer
            auto translated = TranslationTransform<DIM>(esn.States(), collect);

            auto eval_lag = [&](auto& readout, size_t lag)
            {
                size_t valid = collect - lag;
                size_t tr = static_cast<size_t>(valid * 0.7);
                size_t te = valid - tr;
                if (tr == 0 || te == 0) return;

                std::vector<float> targets(valid);
                for (size_t t = 0; t < valid; ++t)
                    targets[t] = inputs[warmup + t];

                const float* lagged = translated.data() + lag * FEATURES;

                readout.Train(lagged, targets.data(), tr, FEATURES);
                double r2 = readout.R2(lagged + tr * FEATURES, targets.data() + tr, te);
                if (r2 > 0.0) r2_sum[lag] += r2;
            };

            for (size_t lag = 1; lag <= MAX_LAG && lag < collect; ++lag)
            {
                if (readout_type_ == ReadoutType::Ridge)
                { RidgeRegression r; eval_lag(r, lag); }
                else
                { LinearReadout r; eval_lag(r, lag); }
            }
        }

        // Average and display
        double n = static_cast<double>(Seeds().size());
        double mc_total = 0.0;

        for (size_t lag : display_lags)
        {
            double r2 = (lag <= MAX_LAG) ? r2_sum[lag] / n : 0.0;
            mc_total += r2;
            std::cout << "  " << std::setw(4) << lag << " | "
                      << std::fixed << std::setprecision(4) << std::setw(6) << r2 << "\n";
        }

        // Full MC sum (all lags, not just displayed)
        double mc_full = 0.0;
        for (size_t lag = 1; lag <= MAX_LAG; ++lag)
            mc_full += r2_sum[lag] / n;

        std::cout << "\n  MC (displayed lags): " << std::setprecision(1) << mc_total;
        std::cout << "\n  MC (all lags 1-" << MAX_LAG << "): " << mc_full << "\n";
    }

private:
    ReadoutType readout_type_;
    const ReservoirConfig* config_;

    static std::vector<uint64_t> Seeds() { return {42, 1042, 2042}; }

    void PrintHeader(size_t warmup, size_t collect) const
    {
        const char* rn = (readout_type_ == ReadoutType::Ridge) ? "Ridge" : "Linear";
        std::cout << "=== Memory Capacity Profile (" << rn << " Readout, full translation, 3-seed avg) ===\n";
        std::cout << "Seeds: {42,1042,2042} | Alpha: 1.0 | Leak: 1.0"
                  << " | SR: per-DIM default | Input scaling: per-DIM default\n";
        std::cout << "DIM=" << DIM << "  N=" << N << "  Features=" << FEATURES
                  << "  Warmup: " << warmup << " | Collect: " << collect
                  << " | MC=sum R2 lags 1-" << MAX_LAG << "\n\n";

        std::cout << "   Lag |     R2\n";
        std::cout << "  -----+-------\n";
    }
};
