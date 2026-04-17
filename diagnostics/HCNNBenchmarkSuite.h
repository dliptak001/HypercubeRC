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
/// Runs Mackey-Glass and NARMA-10 at DIM 5-8 using the HCNN readout only,
/// with the HRCCNN baseline config from `hcnn_presets::HRCCNNBaseline<DIM>()`.
/// Skips MemoryCapacity.  Does **not** compute Ridge raw or Ridge translated
/// — the Ridge data path is completely bypassed, so wall time is purely
/// HCNN training time + reservoir collection.  NRMSE is reported in
/// absolute terms, with no comparison percentages.
///
/// For Mackey-Glass the prediction horizon defaults to **h=84**, the
/// Jaeger & Haas 2004 "hard" chaotic-prediction benchmark (tau=17, MG
/// delay differential equation).  NARMA-10 has no tunable horizon; the
/// task is intrinsically 1-step nonlinear autoregressive.
///
/// This suite is parallel to (not a replacement for) `BenchmarkSuite`.
/// Use `BenchmarkSuite` for the canonical Ridge vs HCNN comparison plus
/// MC; use this one when you only care about HCNN performance at a
/// specific MG horizon and want zero Ridge contamination.
struct HCNNBenchmarkSuite
{
    /// @param mg_horizon       MG prediction horizon in steps (default 84).
    /// @param output_fraction  Reservoir output fraction (default 1.0).
    /// @param config           Optional reservoir config override.
    static void RunAll(size_t mg_horizon = 84,
                       float output_fraction = 1.0f,
                       const ReservoirConfig* config = nullptr)
    {
        std::cout << "=== HypercubeRC HCNN-Only Benchmark Suite ===\n\n";
        std::cout << "Runs Mackey-Glass and NARMA-10 at DIM 5-8 using the HCNN readout\n";
        std::cout << "only.  Ridge raw / Ridge translated / MC are fully skipped -- the\n";
        std::cout << "wall time and output columns are pure HCNN.  HCNN uses the HRCCNN\n";
        std::cout << "baseline config from docs/HRCCNNBaselineConfig.md:\n\n";
        std::cout << "  nl=1  ch=8  FLATTEN  ep=2000  lr_max=0.0015  bs = 1 << (DIM-1)\n\n";
        std::cout << "Mackey-Glass horizon: h=" << mg_horizon
                  << "   (Jaeger & Haas 2004 standard = 84)\n";
        std::cout << "Output fraction: " << static_cast<int>(output_fraction * 100) << "%\n\n";

        std::cout << "--- Mackey-Glass h=" << mg_horizon
                  << " (NRMSE, lower is better) ---\n";
        std::cout << "  DIM |     N |    NRMSE |  time(s)\n";
        std::cout << "  ----+-------+----------+---------\n" << std::flush;
        /*RunAndPrintMG<5>(mg_horizon, output_fraction, config);
        RunAndPrintMG<6>(mg_horizon, output_fraction, config);
        RunAndPrintMG<7>(mg_horizon, output_fraction, config);
        RunAndPrintMG<8>(mg_horizon, output_fraction, config);*/

        std::cout << "\n--- NARMA-10 (NRMSE, lower is better) ---\n";
        std::cout << "  DIM |     N |    NRMSE |  time(s)\n";
        std::cout << "  ----+-------+----------+---------\n" << std::flush;
        //RunAndPrintNARMA<5>(output_fraction, config);
        //RunAndPrintNARMA<6>(output_fraction, config);
        RunAndPrintNARMA<7>(output_fraction, config);
        RunAndPrintNARMA<8>(output_fraction, config);
        RunAndPrintNARMA<9>(output_fraction, config);
        RunAndPrintNARMA<10>(output_fraction, config);
        /*RunAndPrintNARMA<11>(output_fraction, config);
        RunAndPrintNARMA<12>(output_fraction, config);
        RunAndPrintNARMA<13>(output_fraction, config);
        RunAndPrintNARMA<14>(output_fraction, config);*/
    }

private:
    template <size_t DIM>
    static constexpr size_t Warmup()   { return (1ULL << DIM) < 256 ? 200 : 500; }

    template <size_t DIM>
    static constexpr size_t Collect()  { return 18 * (1ULL << DIM); }

    /// Emit the "DIM | N |" prefix and flush so the user sees which row
    /// is actively training.  The NRMSE and time cells are appended by
    /// FinishRow() when training completes, forming a single line.
    template <size_t DIM>
    static void StartRow()
    {
        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(5) << (1ULL << DIM)
                  << " |" << std::flush;
    }

    /// Complete the row started by StartRow() with NRMSE and wall-time.
    static void FinishRow(double nrmse, double elapsed_s)
    {
        std::cout << " " << std::fixed << std::setprecision(6) << std::setw(8) << nrmse
                  << " | " << std::setprecision(2) << std::setw(7) << elapsed_s
                  << "\n" << std::flush;
    }

    /// HCNN-only Mackey-Glass run.  Builds the MG trajectory, drives the
    /// reservoir once, trains the HCNN readout, reports test NRMSE.  Ridge
    /// never touches the data.
    template <size_t DIM>
    static void RunAndPrintMG(size_t horizon, float output_fraction,
                              const ReservoirConfig* config)
    {
        StartRow<DIM>();

        constexpr size_t warmup  = Warmup<DIM>();
        constexpr size_t collect = Collect<DIM>();
        const size_t tr = static_cast<size_t>(collect * 0.7);
        const size_t te = collect - tr;

        auto series = GenerateMackeyGlass(warmup + collect + horizon + 20);
        Normalize(series);
        std::vector<float> targets(collect);
        for (size_t t = 0; t < collect; ++t)
            targets[t] = series[warmup + t + horizon];

        ReservoirConfig cfg = config ? *config : ReservoirConfig{};
        cfg.seed            = hcnn_presets::MackeyGlass<DIM>().reservoir.seed;
        cfg.output_fraction = 1.0f;
        if (output_fraction != 1.0f) cfg.output_fraction = output_fraction;

        ESN<DIM> esn(cfg, ReadoutType::HCNN);
        esn.Warmup(series.data(), warmup);
        esn.Run(series.data() + warmup, collect);

        auto t0 = std::chrono::steady_clock::now();
        esn.Train(targets.data(), tr, hcnn_presets::HRCCNNBaseline<DIM>());
        auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        FinishRow(esn.NRMSE(targets.data(), tr, te), elapsed);
    }

    /// HCNN-only NARMA-10 run.  Generates the NARMA input/target stream,
    /// drives the reservoir once, trains the HCNN readout, reports test
    /// NRMSE.  Ridge never touches the data.
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

        // NARMA inputs live in [0, 0.5]; rescale to [-1, +1] for the reservoir.
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
