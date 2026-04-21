#pragma once

#include <iostream>
#include <iomanip>
#include <utility>
#include "NARMA10.h"

/// @brief Unified NARMA-10 benchmark suite across configurable DIM ranges.
struct BenchmarkSuite
{
    static void RunAll(BenchmarkMode mode = BenchmarkMode::RidgeOnly,
                       float output_fraction = 1.0f,
                       const ReservoirConfig* config = nullptr)
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
        std::cout << "Output fraction: " << static_cast<int>(output_fraction * 100) << "%\n\n";

        const bool show_ridge = (mode == BenchmarkMode::RidgeOnly || mode == BenchmarkMode::Both);
        const bool show_hcnn  = (mode == BenchmarkMode::HCNNOnly  || mode == BenchmarkMode::Both);

        std::cout << "  DIM  -- hypercube dimension; the reservoir has N = 2^DIM neurons\n";
        if (show_ridge) std::cout << "  Ridge -- closed-form readout on reservoir states\n";
        if (show_hcnn) {
            std::cout << "  HCNN  -- HypercubeCNN readout (learned convolution)\n";
            std::cout << "           nl=1 ch=8 FLAT ep=2000 lr=0.0015 bs=1<<(DIM-1)\n";
        }
        std::cout << "\n";

        std::cout << "--- NARMA-10 (NRMSE, lower is better) ---\n";
        std::cout << "Nonlinear autoregressive benchmark requiring both memory (10-step\n";
        std::cout << "history) and nonlinear computation (product terms).\n\n";

        PrintHeader(mode);

        switch (mode) {
            case BenchmarkMode::RidgeOnly:
            case BenchmarkMode::Both:
                RunDIM<5>(mode, output_fraction, config);
                RunDIM<6>(mode, output_fraction, config);
                RunDIM<7>(mode, output_fraction, config);
                RunDIM<8>(mode, output_fraction, config);
                break;
            case BenchmarkMode::HCNNOnly:
                RunDIM<7>(mode, output_fraction, config);
                RunDIM<8>(mode, output_fraction, config);
                RunDIM<9>(mode, output_fraction, config);
                RunDIM<10>(mode, output_fraction, config);
                break;
        }
    }

private:
    static void PrintHeader(BenchmarkMode mode)
    {
        const bool show_ridge = (mode == BenchmarkMode::RidgeOnly || mode == BenchmarkMode::Both);
        const bool show_hcnn  = (mode == BenchmarkMode::HCNNOnly  || mode == BenchmarkMode::Both);

        std::cout << "  DIM |     N";
        if (show_ridge) std::cout << " |     Ridge";
        if (show_hcnn)  std::cout << " |      HCNN";
        if (show_ridge && show_hcnn) std::cout << "            ";
        if (show_hcnn)  std::cout << " |  time(s)";
        std::cout << "\n";

        std::cout << "  ----+------";
        if (show_ridge) std::cout << "-+-----------";
        if (show_hcnn)  std::cout << "-+-----------";
        if (show_ridge && show_hcnn) std::cout << "------------";
        if (show_hcnn)  std::cout << "-+---------";
        std::cout << "\n" << std::flush;
    }

    template <size_t DIM>
    static void RunDIM(BenchmarkMode mode, float output_fraction, const ReservoirConfig* config)
    {
        // Print DIM|N prefix and flush so the user sees which row is training.
        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(4) << (1ULL << DIM) << " |" << std::flush;

        NARMA10<DIM> narma(mode, config, output_fraction);
        auto r = narma.Run();

        const bool show_ridge = (mode == BenchmarkMode::RidgeOnly || mode == BenchmarkMode::Both);
        const bool show_hcnn  = (mode == BenchmarkMode::HCNNOnly  || mode == BenchmarkMode::Both);

        if (show_ridge)
            std::cout << " " << std::fixed << std::setprecision(6) << std::setw(9) << r.nrmse_ridge;

        if (show_hcnn) {
            std::cout << " | " << std::fixed << std::setprecision(6) << std::setw(9) << r.nrmse_hcnn;
            if (show_ridge)
                std::cout << " (" << std::showpos << std::setprecision(1) << std::setw(5)
                          << r.pct_change_hcnn << "%" << std::noshowpos << ")";
            std::cout << " | " << std::setprecision(2) << std::setw(7) << r.hcnn_time_s;
        }

        std::cout << "\n" << std::flush;
    }
};
