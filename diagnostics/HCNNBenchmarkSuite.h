#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../ESN.h"
#include "../readout/HCNNPresets.h"
#include "SignalGenerators.h"

/// @file HCNNBenchmarkSuite.h
/// @brief HCNN-only variant of the HypercubeRC benchmark suite.
///
/// Runs NARMA-10 at DIM 5-8 using the HCNN readout only, with the HRCCNN
/// baseline config from `hcnn_presets::HRCCNNBaseline<DIM>()`.  Ridge is
/// completely bypassed, so wall time is purely HCNN training + reservoir
/// collection.  NRMSE is reported in absolute terms.
///
/// This suite is parallel to (not a replacement for) `BenchmarkSuite`.
struct HCNNBenchmarkSuite
{
    /// @param output_fraction  Reservoir output fraction (default 1.0).
    /// @param config           Optional reservoir config override.
    static void RunAll(float output_fraction = 1.0f,
                       const ReservoirConfig* config = nullptr)
    {
        std::cout << "=== HypercubeRC HCNN-Only Benchmark Suite ===\n\n";
        std::cout << "Runs NARMA-10 at DIM 5-8 using the HCNN readout only.  Ridge is\n";
        std::cout << "fully skipped -- wall time and output are pure HCNN.  Config:\n\n";
        std::cout << "  nl=1  ch=8  FLATTEN  ep=2000  lr_max=0.0015  bs = 1 << (DIM-1)\n\n";
        std::cout << "Output fraction: " << static_cast<int>(output_fraction * 100) << "%\n\n";

        std::cout << "--- NARMA-10 (NRMSE, lower is better) ---\n";
        std::cout << "  DIM |     N |    NRMSE |  time(s)\n";
        std::cout << "  ----+-------+----------+---------\n" << std::flush;
        RunAndPrintNARMA<7>(output_fraction, config);
        RunAndPrintNARMA<8>(output_fraction, config);
        RunAndPrintNARMA<9>(output_fraction, config);
        RunAndPrintNARMA<10>(output_fraction, config);
    }

private:
    template <size_t DIM>
    static constexpr size_t Warmup()   { return (1ULL << DIM) < 256 ? 200 : 500; }

    template <size_t DIM>
    static constexpr size_t Collect()  { return 18 * (1ULL << DIM); }

    template <size_t DIM>
    static void StartRow()
    {
        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(5) << (1ULL << DIM)
                  << " |" << std::flush;
    }

    static void FinishRow(double nrmse, double elapsed_s)
    {
        std::cout << " " << std::fixed << std::setprecision(6) << std::setw(8) << nrmse
                  << " | " << std::setprecision(2) << std::setw(7) << elapsed_s
                  << "\n" << std::flush;
    }

    template <size_t DIM>
    static void RunAndPrintNARMA(float output_fraction, const ReservoirConfig* config)
    {
        StartRow<DIM>();

        constexpr size_t warmup  = Warmup<DIM>();
        constexpr size_t collect = Collect<DIM>();
        const size_t tr = static_cast<size_t>(collect * 0.7);
        const size_t te = collect - tr;

        const uint64_t seed = hcnn_presets::NARMA10<DIM>().reservoir.seed;
        auto [u, y] = GenerateNARMA10(seed + 99, warmup + collect);

        std::vector<float> ri(warmup + collect);
        for (size_t t = 0; t < ri.size(); ++t)
            ri[t] = u[t] * 4.0f - 1.0f;

        std::vector<float> targets(collect);
        for (size_t t = 0; t < collect; ++t)
            targets[t] = y[warmup + t];

        ReservoirConfig cfg = config ? *config : ReservoirConfig{};
        cfg.seed            = seed;
        cfg.output_fraction = 1.0f;
        if (output_fraction != 1.0f) cfg.output_fraction = output_fraction;

        ESN<DIM> esn(cfg, ReadoutType::HCNN);
        esn.Warmup(ri.data(), warmup);
        esn.Run(ri.data() + warmup, collect);

        auto t0 = std::chrono::steady_clock::now();
        esn.Train(targets.data(), tr, hcnn_presets::HRCCNNBaseline<DIM>());
        auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        FinishRow(esn.NRMSE(targets.data(), tr, te), elapsed);
    }
};
