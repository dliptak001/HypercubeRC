#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstddef>
#include "../ESN.h"

/// @brief Diagnostic: Memory capacity profile across lags 1-50.
///
/// For each lag L, fits a readout from translated reservoir states (2.5N features)
/// to the continuous input value from L steps ago: target[t] = input[t - L].
/// Reports per-lag R2 at selected display lags plus total MC (sum of R2 over all
/// lags 1-50).
///
/// NOTE: This diagnostic uses full translation (2.5N features), which yields higher MC
/// than the standard metric (raw N states). The main.cpp cascade benchmark reports both
/// raw and full translation side by side.
///
/// NOTE: Per-lag readouts are created externally because MC trains 50
/// independent readouts (one per lag), which doesn't fit the single-readout
/// ESN pipeline.
template <size_t DIM>
class MemoryCapacityProfile
{
    static constexpr size_t N = 1ULL << DIM;
    static constexpr size_t MAX_LAG = 50;

public:
    MemoryCapacityProfile(ReadoutType readout_type = ReadoutType::Ridge,
                          const ReservoirConfig* config = nullptr,
                          float output_fraction = 1.0f)
        : readout_type_(readout_type), config_(config), output_fraction_(output_fraction)
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

            ReservoirConfig cfg = config_ ? *config_ : ReservoirConfig{};
            cfg.seed = seed;
            if (output_fraction_ != 1.0f)
                cfg.output_fraction = output_fraction_;
            ESN<DIM> esn(cfg, readout_type_, FeatureMode::Translated);
            esn.Warmup(inputs.data(), warmup);
            esn.Run(inputs.data() + warmup, collect);
            esn.EnsureFeatures();

            size_t nf = esn.NumFeatures();
            const float* features = esn.Features();

            auto eval_lag = [&](auto& readout, size_t lag)
            {
                size_t valid = collect - lag;
                size_t tr = static_cast<size_t>(valid * 0.7);
                size_t te = valid - tr;
                if (tr == 0 || te == 0) return;

                std::vector<float> targets(valid);
                for (size_t t = 0; t < valid; ++t)
                    targets[t] = inputs[warmup + t];

                const float* lagged = features + lag * nf;

                readout.Train(lagged, targets.data(), tr, nf);
                double r2 = readout.R2(lagged + tr * nf, targets.data() + tr, te);
                if (r2 > 0.0) r2_sum[lag] += r2;
            };

            for (size_t lag = 1; lag <= MAX_LAG && lag < collect; ++lag)
            {
                RidgeRegression r; eval_lag(r, lag);
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
    float output_fraction_;

    static constexpr uint64_t DefaultSeed()
    {
        if constexpr (DIM == 5) return 7778726955320718972ULL;
        else if constexpr (DIM == 6) return 17341644007929035161ULL;
        else if constexpr (DIM == 7) return 11931814417146401966ULL;
        else if constexpr (DIM == 8) return 14376161041117039141ULL;
        else return 42;
    }

    static std::vector<uint64_t> Seeds()
    {
        if (single_seed) return {single_seed};
        return {DefaultSeed()};
    }

public:
    static inline thread_local uint64_t single_seed = 0;  // non-zero = use only this seed

private:

    void PrintHeader(size_t warmup, size_t collect) const
    {
        const char* rn = "Ridge";
        std::cout << "=== Memory Capacity Profile (" << rn << " Readout, full translation) ===\n";
        std::cout << "Seed: " << DefaultSeed() << " | Alpha: 1.0 | Leak: 1.0"
                  << " | SR: 0.90 | Input scaling: 0.02\n";
        float frac = output_fraction_;
        size_t M = std::max<size_t>(1, static_cast<size_t>(std::round(N * frac)));
        std::cout << "DIM=" << DIM << "  N=" << N << "  Outputs=" << M
                  << "  Features=" << TranslationFeatureCountSelected(M)
                  << "  Warmup: " << warmup << " | Collect: " << collect
                  << " | MC=sum R2 lags 1-" << MAX_LAG << "\n\n";

        std::cout << "   Lag |     R2\n";
        std::cout << "  -----+-------\n";
    }
};
