#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstddef>
#include "../ESN.h"
#include "../Reservoir.h"
#include "../readout/LinearReadout.h"

/// @brief Diagnostic: Total memory capacity (raw features, LinearReadout).
///
/// Measures how many past inputs the reservoir can reconstruct via a linear
/// readout. For each lag L in [1, max_lag], trains a LinearReadout to predict
/// input[t - L] from raw reservoir states, and computes R². Total MC is the
/// sum of R² over all lags.
///
/// Higher is better. Theoretical maximum = N (number of neurons).
///
/// This is the lightweight version — raw N features, LinearReadout only.
/// See MemoryCapacityProfile for the full translation-layer version with
/// per-lag breakdown.
template <size_t DIM>
class MemoryCapacity
{
    static constexpr size_t N = 1ULL << DIM;

public:
    struct Result
    {
        double mc_total;
    };

    MemoryCapacity(size_t max_lag = 50,
                   const ReservoirConfig* config = nullptr,
                   float output_fraction = 1.0f)
        : max_lag_(max_lag), config_(config), output_fraction_(output_fraction)
    {
    }

    Result Run()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        double s_mc = 0.0;

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
            ESN<DIM> esn(cfg, ReadoutType::Linear);
            esn.Warmup(inputs.data(), warmup);
            esn.Run(inputs.data() + warmup, collect);

            auto selected = esn.SelectedStates();
            size_t M = esn.NumOutputVerts();
            double mc = 0.0;
            size_t num_lags = std::min(max_lag_, collect - 1);

            #pragma omp parallel for reduction(+:mc) schedule(dynamic)
            for (size_t lag = 1; lag <= num_lags; ++lag)
            {
                size_t valid = collect - lag;
                std::vector<float> tgt(valid);
                for (size_t t = 0; t < valid; ++t)
                    tgt[t] = inputs[warmup + t];

                const float* vs = selected.data() + lag * M;
                size_t tr = static_cast<size_t>(valid * 0.7);
                size_t te = valid - tr;
                if (tr == 0 || te == 0) continue;

                LinearReadout lr;
                lr.Train(vs, tgt.data(), tr, M);
                double r2 = lr.R2(vs + tr * M, tgt.data() + tr, te);
                if (r2 > 0.0) mc += r2;
            }
            s_mc += mc;
        }

        return {s_mc / static_cast<double>(Seeds().size())};
    }

    void RunAndPrint()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;
        PrintHeader(warmup, collect);

        auto r = Run();

        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(5) << N
                  << " | " << std::fixed << std::setprecision(1) << std::setw(5) << r.mc_total << "\n";
    }

private:
    size_t max_lag_;
    const ReservoirConfig* config_;
    float output_fraction_;

    static constexpr uint64_t DefaultSeed()
    {
        if constexpr (DIM == 5) return 7778726955320718972ULL;
        else if constexpr (DIM == 6) return 17341644007929035161ULL;
        else if constexpr (DIM == 7) return 11931814417146401966ULL;
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
        std::cout << "=== Memory Capacity (LinearReadout, raw features) ===\n";
        std::cout << "Seed: " << DefaultSeed() << " | Alpha: 1.0 | Leak: 1.0"
                  << " | SR: 0.90 | Input scaling: 0.02\n";
        std::cout << "Warmup: " << warmup << " | Collect: " << collect
                  << " | MC = sum R2 lags 1-" << max_lag_ << "\n\n";

        std::cout << "  DIM |     N |    MC\n";
        std::cout << "  ----+-------+------\n";
    }
};
