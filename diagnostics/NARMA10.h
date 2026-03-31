#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstddef>
#include <cmath>
#include "../ESN.h"
#include "../Reservoir.h"
#include "../TranslationLayer.h"
#include "../SignalGenerators.h"
#include "../readout/LinearReadout.h"
#include "../readout/RidgeRegression.h"

/// @brief Diagnostic: NARMA-10 nonlinear benchmark.
///
/// NARMA-10 (Nonlinear AutoRegressive Moving Average, order 10) is the standard
/// RC benchmark for combined memory + nonlinear computation. The target is:
///
///   y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1
///
/// Reports NRMSE for both raw (N) and full translation (2.5N) features,
/// 3-seed average. Readout type (Linear or Ridge) is configurable.
/// Standard ESN NRMSE on NARMA-10: 0.2-0.4 (lower is better).
template <size_t DIM>
class NARMA10
{
    static constexpr size_t N = 1ULL << DIM;

public:
    struct Result
    {
        double nrmse_raw;
        double nrmse_full;
        double pct_change;  // % change from raw to full
    };

    NARMA10(ReadoutType readout_type = ReadoutType::Linear,
            const ReservoirConfig* config = nullptr, float output_fraction = 1.0f)
        : readout_type_(readout_type), config_(config), output_fraction_(output_fraction)
    {
    }

    /// @brief Run the benchmark and return results without printing.
    Result Run()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        double s_nrmse_raw = 0.0, s_nrmse_full = 0.0;

        auto eval = [](auto& readout, const float* features, const float* targets,
                       size_t tr, size_t te, size_t nf)
        {
            readout.Train(features, targets, tr, nf);
            return ComputeNRMSE(readout, features + tr * nf, targets + tr, te, nf);
        };

        for (uint64_t seed : Seeds())
        {
            auto [u, y] = GenerateNARMA10(seed + 99, warmup + collect);

            // Scale inputs to [-1, 1] for the reservoir (NARMA uses [0, 0.5])
            std::vector<float> ri(warmup + collect);
            for (size_t t = 0; t < ri.size(); ++t)
                ri[t] = u[t] * 4.0f - 1.0f;

            std::vector<float> targets(collect);
            for (size_t t = 0; t < collect; ++t)
                targets[t] = y[warmup + t];

            size_t tr = static_cast<size_t>(collect * 0.7);
            size_t te = collect - tr;

            // Raw features
            {
                ReservoirConfig cfg = config_ ? *config_ : ReservoirConfig{};
                cfg.seed = seed;
                if (output_fraction_ != 1.0f)
                    cfg.output_fraction = output_fraction_;
                ESN<DIM> esn(cfg, readout_type_);
                esn.Warmup(ri.data(), warmup);
                esn.Run(ri.data() + warmup, collect);
                auto selected = esn.SelectedStates();
                size_t M = esn.NumOutputVerts();
                if (readout_type_ == ReadoutType::Ridge)
                { RidgeRegression r; s_nrmse_raw += eval(r, selected.data(), targets.data(), tr, te, M); }
                else
                { LinearReadout r; s_nrmse_raw += eval(r, selected.data(), targets.data(), tr, te, M); }
            }

            // Full translation
            {
                ReservoirConfig cfg = config_ ? *config_ : ReservoirConfig{};
                cfg.seed = seed;
                if (output_fraction_ != 1.0f)
                    cfg.output_fraction = output_fraction_;
                ESN<DIM> esn(cfg, readout_type_);
                esn.Warmup(ri.data(), warmup);
                esn.Run(ri.data() + warmup, collect);
                size_t M = esn.NumOutputVerts();
                auto translated = TranslationTransformSelected<DIM>(esn.States(), collect,
                                                                     esn.OutputStride(), M);
                size_t nf = TranslationFeatureCountSelected(M);
                if (readout_type_ == ReadoutType::Ridge)
                { RidgeRegression r; s_nrmse_full += eval(r, translated.data(), targets.data(), tr, te, nf); }
                else
                { LinearReadout r; s_nrmse_full += eval(r, translated.data(), targets.data(), tr, te, nf); }
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
                  << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r.nrmse_raw
                  << " | " << std::setprecision(3) << std::setw(7) << r.nrmse_full
                  << " (" << std::showpos << std::setprecision(1) << std::setw(5) << r.pct_change
                  << "%" << std::noshowpos << ")\n";
    }

private:
    ReadoutType readout_type_;
    const ReservoirConfig* config_;
    float output_fraction_;

    static std::vector<uint64_t> Seeds()
    {
        if (single_seed) return {single_seed};
        return {42, 1042, 2042};
    }

public:
    static inline uint64_t single_seed = 0;  // non-zero = use only this seed

private:
    void PrintHeader(size_t warmup, size_t collect) const
    {
        const char* rn = (readout_type_ == ReadoutType::Ridge) ? "Ridge" : "Linear";
        std::cout << "=== NARMA-10 (" << rn << " Readout, " << Seeds().size() << "-seed avg, raw vs full translation) ===\n";
        std::cout << "Seeds: {42,1042,2042} | Alpha: 1.0 | Leak: 1.0"
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
