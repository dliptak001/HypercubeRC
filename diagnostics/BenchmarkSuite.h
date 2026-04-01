#pragma once

#include <iostream>
#include <iomanip>
#include "MemoryCapacity.h"
#include "MackeyGlass.h"
#include "NARMA10.h"

/// @brief Per-DIM benchmark runner. Runs MC, MG, and NARMA-10.
template <size_t DIM>
struct BenchmarkDIM
{
    struct Result
    {
        double mc_total;
        typename MackeyGlass<DIM>::Result mg;
        typename NARMA10<DIM>::Result narma;
    };

    static Result Run(const ReservoirConfig* config = nullptr, float output_fraction = 1.0f)
    {
        MemoryCapacity<DIM> mc(50, config, output_fraction);
        MackeyGlass<DIM> mg(1, ReadoutType::Ridge, config, output_fraction);
        NARMA10<DIM> narma(ReadoutType::Ridge, config, output_fraction);
        return {mc.Run().mc_total, mg.Run(), narma.Run()};
    }

    static void PrintMCRow(double mc_total)
    {
        std::cout << "  " << std::setw(3) << DIM
            << "  | " << std::setw(5) << (1ULL << DIM)
            << " | " << std::fixed << std::setprecision(1) << std::setw(5) << mc_total << "\n";
    }

    static void PrintMGRow(const typename MackeyGlass<DIM>::Result& r)
    {
        std::cout << "  " << std::setw(3) << DIM
            << "  | " << std::setw(5) << (1ULL << DIM)
            << " | " << std::fixed << std::setprecision(4) << std::setw(7) << r.nrmse_raw
            << " | " << std::setprecision(4) << std::setw(7) << r.nrmse_full
            << " (" << std::showpos << std::setprecision(1) << std::setw(5) << r.pct_change
            << "%" << std::noshowpos << ")\n";
    }

    static void PrintNARMARow(const typename NARMA10<DIM>::Result& r)
    {
        std::cout << "  " << std::setw(3) << DIM
            << "  | " << std::setw(5) << (1ULL << DIM)
            << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r.nrmse_raw
            << " | " << std::setprecision(3) << std::setw(7) << r.nrmse_full
            << " (" << std::showpos << std::setprecision(1) << std::setw(5) << r.pct_change
            << "%" << std::noshowpos << ")\n";
    }
};

/// @brief Full benchmark suite across DIM 5-8 (library supports 5-12). One call does everything.
struct BenchmarkSuite
{
    static void RunAll(float output_fraction = 1.0f, const ReservoirConfig* config = nullptr)
    {
        std::cout << "=== HypercubeRC Benchmark Suite ===\n\n";
        std::cout << "HypercubeRC is a reservoir computer whose N neurons are arranged on\n";
        std::cout << "a Boolean hypercube -- a DIM-dimensional graph where each vertex is\n";
        std::cout << "addressed by a DIM-bit binary index, and two vertices are neighbors\n";
        std::cout << "if their indices differ by one bit (computed by XOR, no adjacency\n";
        std::cout << "list needed). Input and output are continuous scalar values.\n\n";
        std::cout << "The pipeline: drive the reservoir with a scalar input signal, collect\n";
        std::cout << "the N-dimensional state at each step, then train a linear readout to\n";
        std::cout << "map those states to the target. The reservoir's weights are fixed --\n";
        std::cout << "only the readout is learned. Each benchmark uses its optimal seed\n";
        std::cout << "per DIM, selected by 500-seed survey (see docs/SeedSurvey.md).\n";
        std::cout << "Output fraction: " << static_cast<int>(output_fraction * 100) << "%\n\n";
        std::cout << "  DIM  -- hypercube dimension; the reservoir has N = 2^DIM neurons\n";
        std::cout << "  raw  -- readout uses N raw reservoir states\n";
        std::cout << "  full -- readout uses 2.5M features after translation (x, x^2, x*x')\n";
        std::cout << "          the extra nonlinear terms help the linear readout decode\n";
        std::cout << "          information that tanh folds into the state vector\n\n";

        std::cout << "--- Memory Capacity (lags 1-50) ---\n";
        std::cout << "How many past inputs can the reservoir reconstruct? MC = sum of R^2\n";
        std::cout << "over lags 1-50. Higher is better. Theoretical max = N.\n\n";
        std::cout << "  DIM |     N |    MC\n";
        std::cout << "  ----+-------+------\n" << std::flush;
        RunAndPrintMC<5>(output_fraction, config);
        RunAndPrintMC<6>(output_fraction, config);
        RunAndPrintMC<7>(output_fraction, config);
        RunAndPrintMC<8>(output_fraction, config);

        std::cout << "\n--- Mackey-Glass h=1 (NRMSE, lower is better) ---\n";
        std::cout << "One-step prediction of a chaotic time series. Tests how well the\n";
        std::cout << "reservoir tracks complex, deterministic dynamics.\n\n";
        std::cout << "  DIM |     N |    raw  |   full translation\n";
        std::cout << "  ----+-------+---------+-----------------\n" << std::flush;
        RunAndPrintMG<5>(output_fraction, config);
        RunAndPrintMG<6>(output_fraction, config);
        RunAndPrintMG<7>(output_fraction, config);
        RunAndPrintMG<8>(output_fraction, config);

        std::cout << "\n--- NARMA-10 (NRMSE, lower is better) ---\n";
        std::cout << "Nonlinear autoregressive benchmark requiring both memory (10-step\n";
        std::cout << "history) and nonlinear computation (product terms). This is where\n";
        std::cout << "the translation layer has the biggest impact.\n\n";
        std::cout << "  DIM |     N |    raw  |   full translation\n";
        std::cout << "  ----+-------+---------+-----------------\n" << std::flush;
        RunAndPrintNARMA<5>(output_fraction, config);
        RunAndPrintNARMA<6>(output_fraction, config);
        RunAndPrintNARMA<7>(output_fraction, config);
        RunAndPrintNARMA<8>(output_fraction, config);
    }

private:
    template <size_t DIM>
    static void RunAndPrintMC(float output_fraction, const ReservoirConfig* config)
    {
        MemoryCapacity<DIM> mc(50, config, output_fraction);
        BenchmarkDIM<DIM>::PrintMCRow(mc.Run().mc_total);
        std::cout << std::flush;
    }

    template <size_t DIM>
    static void RunAndPrintMG(float output_fraction, const ReservoirConfig* config)
    {
        MackeyGlass<DIM> mg(1, ReadoutType::Ridge, config, output_fraction);
        BenchmarkDIM<DIM>::PrintMGRow(mg.Run());
        std::cout << std::flush;
    }

    template <size_t DIM>
    static void RunAndPrintNARMA(float output_fraction, const ReservoirConfig* config)
    {
        NARMA10<DIM> narma(ReadoutType::Ridge, config, output_fraction);
        BenchmarkDIM<DIM>::PrintNARMARow(narma.Run());
        std::cout << std::flush;
    }
};
