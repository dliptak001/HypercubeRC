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

enum class BenchmarkMode { RidgeOnly, HCNNOnly, Both };

/// @brief Diagnostic: NARMA-10 nonlinear benchmark.
///
/// NARMA-10 (Nonlinear AutoRegressive Moving Average, order 10) is the standard
/// RC benchmark for combined memory + nonlinear computation. The target is:
///
///   y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1
///
/// Reports NRMSE and wall-clock timing for Ridge, HCNN, or both.
template <size_t DIM>
class NARMA10
{
    static constexpr size_t N = 1ULL << DIM;

public:
    struct Result
    {
        double nrmse_ridge;      // -1.0 if not run
        double nrmse_hcnn;       // -1.0 if not run
        double pct_change_hcnn;  // % change ridge -> hcnn (0.0 if either missing)
        double ridge_time_s;     // -1.0 if not run
        double hcnn_time_s;      // -1.0 if not run
    };

    NARMA10(BenchmarkMode mode = BenchmarkMode::RidgeOnly,
            const ReservoirConfig* config = nullptr,
            float output_fraction = 1.0f,
            const HCNNReadoutConfig& hcnn_config = BenchmarkCNNConfig())
        : mode_(mode), config_(config), output_fraction_(output_fraction),
          hcnn_config_(hcnn_config)
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

        const bool run_ridge = (mode_ == BenchmarkMode::RidgeOnly || mode_ == BenchmarkMode::Both);
        const bool run_hcnn  = (mode_ == BenchmarkMode::HCNNOnly  || mode_ == BenchmarkMode::Both);

        double s_nrmse_ridge = 0.0, s_nrmse_hcnn = 0.0;
        double s_time_ridge  = 0.0, s_time_hcnn  = 0.0;

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
            if (output_fraction_ != 1.0f)
                cfg.output_fraction = output_fraction_;

            if (run_ridge) {
                ESN<DIM> esn(cfg, ReadoutType::Ridge);
                esn.Warmup(ri.data(), warmup);
                esn.Run(ri.data() + warmup, collect);

                auto t0 = std::chrono::steady_clock::now();
                esn.Train(targets.data(), tr);
                auto t1 = std::chrono::steady_clock::now();

                s_nrmse_ridge += esn.NRMSE(targets.data(), tr, te);
                s_time_ridge  += std::chrono::duration<double>(t1 - t0).count();
            }

            if (run_hcnn) {
                ReservoirConfig hcnn_cfg = cfg;
                hcnn_cfg.output_fraction = 1.0f;

                ESN<DIM> esn(hcnn_cfg, ReadoutType::HCNN);
                esn.Warmup(ri.data(), warmup);
                esn.Run(ri.data() + warmup, collect);

                auto t0 = std::chrono::steady_clock::now();
                esn.Train(targets.data(), tr, hcnn_config_);
                auto t1 = std::chrono::steady_clock::now();

                s_nrmse_hcnn += esn.NRMSE(targets.data(), tr, te);
                s_time_hcnn  += std::chrono::duration<double>(t1 - t0).count();
            }
        }

        double n = static_cast<double>(Seeds().size());
        double nrmse_ridge = run_ridge ? s_nrmse_ridge / n : -1.0;
        double nrmse_hcnn  = run_hcnn  ? s_nrmse_hcnn  / n : -1.0;
        double time_ridge  = run_ridge ? s_time_ridge  / n : -1.0;
        double time_hcnn   = run_hcnn  ? s_time_hcnn   / n : -1.0;
        double pct_hcnn    = (run_ridge && run_hcnn && nrmse_ridge > 1e-12)
                                 ? 100.0 * (nrmse_hcnn - nrmse_ridge) / nrmse_ridge : 0.0;

        return {nrmse_ridge, nrmse_hcnn, pct_hcnn, time_ridge, time_hcnn};
    }

    static uint64_t DefaultSeed()
    {
        return hcnn_presets::NARMA10<DIM>().reservoir.seed;
    }

    static inline thread_local uint64_t single_seed = 0;

private:
    BenchmarkMode mode_;
    const ReservoirConfig* config_;
    float output_fraction_;
    HCNNReadoutConfig hcnn_config_;

    static std::vector<uint64_t> Seeds()
    {
        if (single_seed) return {single_seed};
        return {DefaultSeed()};
    }
};
