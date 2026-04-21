#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../ESN.h"
#include "../readout/HCNNPresets.h"
#include "SignalGenerators.h"

/// @brief Diagnostic: NARMA-10 nonlinear benchmark.
///
/// NARMA-10 (Nonlinear AutoRegressive Moving Average, order 10) is the standard
/// RC benchmark for combined memory + nonlinear computation. The target is:
///
///   y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1
///
/// Reports NRMSE and wall-clock timing for the HCNN readout.
template <size_t DIM>
class NARMA10
{
    static constexpr size_t N = 1ULL << DIM;

public:
    struct Result
    {
        double nrmse_hcnn;
        double hcnn_time_s;
    };

    NARMA10(const ReservoirConfig* config = nullptr,
            const HCNNReadoutConfig& hcnn_config = BenchmarkCNNConfig())
        : config_(config), hcnn_config_(hcnn_config)
    {
    }

    static HCNNReadoutConfig BenchmarkCNNConfig()
    {
        return hcnn_presets::HRCCNNBaseline<DIM>();
    }

    Result Run()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        double s_nrmse = 0.0;
        double s_time  = 0.0;

        for (uint64_t seed : Seeds())
        {
            auto [u, y] = GenerateNARMA10(seed + 99, warmup + collect);

            std::vector<float> ri(warmup + collect);
            for (size_t t = 0; t < ri.size(); ++t)
                ri[t] = u[t] * 4.0f - 1.0f;

            std::vector<float> targets(collect);
            for (size_t t = 0; t < collect; ++t)
                targets[t] = y[warmup + t];

            size_t tr = static_cast<size_t>(collect * 0.7);
            size_t te = collect - tr;

            ReservoirConfig cfg = config_ ? *config_ : ReservoirConfig{};
            cfg.seed = seed;
            cfg.output_fraction = 1.0f;

            ESN<DIM> esn(cfg);
            esn.Warmup(ri.data(), warmup);
            esn.Run(ri.data() + warmup, collect);

            auto t0 = std::chrono::steady_clock::now();
            esn.Train(targets.data(), tr, hcnn_config_);
            auto t1 = std::chrono::steady_clock::now();

            s_nrmse += esn.NRMSE(targets.data(), tr, te);
            s_time  += std::chrono::duration<double>(t1 - t0).count();
        }

        double n = static_cast<double>(Seeds().size());
        return {s_nrmse / n, s_time / n};
    }

    static uint64_t DefaultSeed()
    {
        return hcnn_presets::NARMA10<DIM>().reservoir.seed;
    }

    static inline thread_local uint64_t single_seed = 0;

private:
    const ReservoirConfig* config_;
    HCNNReadoutConfig hcnn_config_;

    static std::vector<uint64_t> Seeds()
    {
        if (single_seed) return {single_seed};
        return {DefaultSeed()};
    }
};
