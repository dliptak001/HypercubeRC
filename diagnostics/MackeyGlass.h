#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstddef>
#include <cmath>
#include "../ESN.h"
#include "SignalGenerators.h"

/// @brief Diagnostic: Mackey-Glass chaotic time series prediction.
///
/// Mackey-Glass delay differential equation (tau=17, n=10, beta=0.2, gamma=0.1)
/// produces low-dimensional chaos. Task: predict x(t+horizon) from reservoir states.
///
/// Reports NRMSE for both raw (N) and full translation (2.5N) features.
/// Readout type (Linear or Ridge) is configurable.
/// Standard ESN NRMSE on MG h=1: 0.01-0.05 (lower is better).
template <size_t DIM>
class MackeyGlass
{
    static constexpr size_t N = 1ULL << DIM;

public:
    struct Result
    {
        double nrmse_raw;
        double nrmse_full;
        double nrmse_hcnn;       // -1.0 if not run
        double pct_change;       // % change raw → full
        double pct_change_hcnn;  // % change raw → hcnn
    };

    MackeyGlass(size_t prediction_horizon = 1, ReadoutType readout_type = ReadoutType::Linear,
                const ReservoirConfig* config = nullptr, float output_fraction = 1.0f,
                bool run_hcnn = false, const CNNReadoutConfig& hcnn_config = BenchmarkCNNConfig())
        : prediction_horizon_(prediction_horizon), readout_type_(readout_type),
          config_(config), output_fraction_(output_fraction),
          run_hcnn_(run_hcnn), hcnn_config_(hcnn_config)
    {
    }

    /// @brief Benchmark-tuned HCNN defaults: larger batch, moderate epochs.
    static CNNReadoutConfig BenchmarkCNNConfig()
    {
        CNNReadoutConfig cfg;
        cfg.epochs = 300;
        cfg.batch_size = 128;
        cfg.lr_max = 0.003f;
        return cfg;
    }

    /// @brief Run the benchmark and return results without printing.
    Result Run()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        double s_nrmse_raw = 0.0, s_nrmse_full = 0.0, s_nrmse_hcnn = 0.0;

        for (uint64_t seed : Seeds())
        {
            auto series = GenerateMackeyGlass(warmup + collect + prediction_horizon_ + 20);
            Normalize(series);

            std::vector<float> targets(collect);
            for (size_t t = 0; t < collect; ++t)
                targets[t] = series[warmup + t + prediction_horizon_];

            size_t tr = static_cast<size_t>(collect * 0.7);
            size_t te = collect - tr;

            ReservoirConfig cfg = config_ ? *config_ : ReservoirConfig{};
            cfg.seed = seed;
            if (output_fraction_ != 1.0f)
                cfg.output_fraction = output_fraction_;

            // Ridge: raw features
            {
                ESN<DIM> esn(cfg, readout_type_, FeatureMode::Raw);
                esn.Warmup(series.data(), warmup);
                esn.Run(series.data() + warmup, collect);
                esn.Train(targets.data(), tr);
                s_nrmse_raw += esn.NRMSE(targets.data(), tr, te);
            }

            // Ridge: full translation
            {
                ESN<DIM> esn(cfg, readout_type_, FeatureMode::Translated);
                esn.Warmup(series.data(), warmup);
                esn.Run(series.data() + warmup, collect);
                esn.Train(targets.data(), tr);
                s_nrmse_full += esn.NRMSE(targets.data(), tr, te);
            }

            // HCNN
            if (run_hcnn_) {
                ReservoirConfig hcnn_cfg = cfg;
                hcnn_cfg.output_fraction = 1.0f;
                ESN<DIM> esn(hcnn_cfg, ReadoutType::HCNN);
                esn.Warmup(series.data(), warmup);
                esn.Run(series.data() + warmup, collect);
                esn.Train(targets.data(), tr, hcnn_config_);
                s_nrmse_hcnn += esn.NRMSE(targets.data(), tr, te);
            }
        }

        double n = static_cast<double>(Seeds().size());
        double nrmse_raw = s_nrmse_raw / n;
        double nrmse_full = s_nrmse_full / n;
        double nrmse_hcnn = run_hcnn_ ? s_nrmse_hcnn / n : -1.0;
        double pct = (nrmse_raw > 1e-12) ? 100.0 * (nrmse_full - nrmse_raw) / nrmse_raw : 0.0;
        double pct_hcnn = (run_hcnn_ && nrmse_raw > 1e-12)
                              ? 100.0 * (nrmse_hcnn - nrmse_raw) / nrmse_raw : 0.0;

        return {nrmse_raw, nrmse_full, nrmse_hcnn, pct, pct_hcnn};
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
    float output_fraction_;
    bool run_hcnn_;
    CNNReadoutConfig hcnn_config_;

    static constexpr uint64_t DefaultSeed()
    {
        if constexpr (DIM == 5) return 11822067163148543833ULL;
        else if constexpr (DIM == 6) return 11459651989651327597ULL;
        else if constexpr (DIM == 7) return 10741866950647888161ULL;
        else if constexpr (DIM == 8) return 2121059498467618174ULL;
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
        const char* rn = (readout_type_ == ReadoutType::Ridge) ? "Ridge" : "Linear";
        std::cout << "=== Mackey-Glass h=" << prediction_horizon_
                  << " (" << rn << " Readout, raw vs full translation) ===\n";
        std::cout << "Seed: " << DefaultSeed() << " | Alpha: 1.0 | Leak: 1.0"
                  << " | SR: 0.90 | Input scaling: 0.02\n";
        float frac = output_fraction_;
        size_t M = std::max<size_t>(1, static_cast<size_t>(std::round(N * frac)));
        std::cout << "Warmup: " << warmup << " | Collect: " << collect
                  << " | Outputs: " << M << "/" << N
                  << " | Features: " << M << " raw, " << TranslationFeatureCountSelected(M) << " translated\n\n";

        std::cout << "  DIM |     N |    raw  |   full\n";
        std::cout << "  ----+-------+---------+-----------------\n";
    }
};
