#pragma once

#include <iostream>
#include <iomanip>
#include "NARMA10.h"

/// @brief Per-DIM benchmark runner. Runs NARMA-10.
template <size_t DIM>
struct BenchmarkDIM
{
    struct Result
    {
        typename NARMA10<DIM>::Result narma;
    };

    static Result Run(const ReservoirConfig* config = nullptr, float output_fraction = 1.0f,
                      bool run_hcnn = false)
    {
        NARMA10<DIM> narma(ReadoutType::Ridge, config, output_fraction, run_hcnn);
        return {narma.Run()};
    }

    static void PrintNARMARow(const typename NARMA10<DIM>::Result& r, bool show_hcnn = false)
    {
        std::cout << "  " << std::setw(3) << DIM
            << "  | " << std::setw(5) << (1ULL << DIM)
            << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r.nrmse_raw
            << " | " << std::setprecision(3) << std::setw(7) << r.nrmse_full
            << " (" << std::showpos << std::setprecision(1) << std::setw(5) << r.pct_change
            << "%" << std::noshowpos << ")";
        if (show_hcnn && r.nrmse_hcnn >= 0.0)
            std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r.nrmse_hcnn
                      << " (" << std::showpos << std::setprecision(1) << std::setw(5) << r.pct_change_hcnn
                      << "%" << std::noshowpos << ")";
        std::cout << "\n";
    }
};

/// @brief Benchmark suite across DIM 5-8 (library supports 5-16). One call does everything.
struct BenchmarkSuite
{
    static void RunAll(float output_fraction = 1.0f, const ReservoirConfig* config = nullptr,
                       bool run_hcnn = false)
    {
        std::cout << "=== HypercubeRC Benchmark Suite ===\n\n";
        std::cout << "HypercubeRC is a reservoir computer whose N neurons are arranged on\n";
        std::cout << "a Boolean hypercube -- a DIM-dimensional graph where each vertex is\n";
        std::cout << "addressed by a DIM-bit binary index, and two vertices are neighbors\n";
        std::cout << "if their indices differ by one bit (computed by XOR, no adjacency\n";
        std::cout << "list needed). Input and output are continuous scalar values.\n\n";
        std::cout << "The pipeline: drive the reservoir with a scalar input signal, collect\n";
        std::cout << "the N-dimensional state at each step, then train a readout to map\n";
        std::cout << "those states to the target. The reservoir's weights are fixed --\n";
        std::cout << "only the readout is learned. Each benchmark uses its optimal seed\n";
        std::cout << "per DIM, selected by 500-seed survey.\n";
        std::cout << "Output fraction: " << static_cast<int>(output_fraction * 100) << "%";
        if (run_hcnn) std::cout << "  [HCNN enabled]";
        std::cout << "\n\n";
        std::cout << "  DIM  -- hypercube dimension; the reservoir has N = 2^DIM neurons\n";
        std::cout << "  raw  -- Ridge readout on N raw reservoir states\n";
        std::cout << "  full -- Ridge readout on 2.5M translated features (x, x^2, x*x')\n";
        if (run_hcnn) {
            std::cout << "  HCNN -- HypercubeCNN readout on N raw states (learned convolution)\n";
            std::cout << "          config: nl=1 ch=8 FLAT ep=2000 lr=0.0015 bs=1<<(DIM-1)"
                      << "  [HRCCNN baseline]\n";
        }
        std::cout << "\n";

        std::cout << "--- NARMA-10 (NRMSE, lower is better) ---\n";
        std::cout << "Nonlinear autoregressive benchmark requiring both memory (10-step\n";
        std::cout << "history) and nonlinear computation (product terms). This is where\n";
        std::cout << "the translation layer has the biggest impact.\n\n";
        std::cout << "  DIM |     N |    raw  |   full translation";
        if (run_hcnn) std::cout << "     |   HCNN";
        std::cout << "\n  ----+-------+---------+-----------------";
        if (run_hcnn) std::cout << "--+------------------";
        std::cout << "\n" << std::flush;
        RunAndPrintNARMA<5>(output_fraction, config, run_hcnn);
        RunAndPrintNARMA<6>(output_fraction, config, run_hcnn);
        RunAndPrintNARMA<7>(output_fraction, config, run_hcnn);
        RunAndPrintNARMA<8>(output_fraction, config, run_hcnn);
    }

private:
    template <size_t DIM>
    static void RunAndPrintNARMA(float output_fraction, const ReservoirConfig* config,
                                 bool run_hcnn = false)
    {
        NARMA10<DIM> narma(ReadoutType::Ridge, config, output_fraction, run_hcnn);
        BenchmarkDIM<DIM>::PrintNARMARow(narma.Run(), run_hcnn);
        std::cout << std::flush;
    }
};
