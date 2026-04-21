#pragma once

#include <iostream>
#include <iomanip>
#include "NARMA10.h"

/// @brief Unified NARMA-10 benchmark suite across configurable DIM ranges.
struct BenchmarkSuite
{
    static void RunAll(const ReservoirConfig* config = nullptr)
    {
        std::cout << "=== HypercubeRC Benchmark Suite ===\n\n";
        std::cout << "HypercubeRC is a reservoir computer whose N neurons are arranged on\n";
        std::cout << "a Boolean hypercube -- a DIM-dimensional graph where each vertex is\n";
        std::cout << "addressed by a DIM-bit binary index, and two vertices are neighbors\n";
        std::cout << "if their indices differ by one bit (computed by XOR, no adjacency\n";
        std::cout << "list needed). Input and output are continuous scalar values.\n\n";
        std::cout << "The pipeline: drive the reservoir with a scalar input signal, collect\n";
        std::cout << "the N-dimensional state at each step, then train the HCNN readout to\n";
        std::cout << "map those states to the target. The reservoir's weights are fixed --\n";
        std::cout << "only the readout is learned. Each benchmark uses its optimal seed\n";
        std::cout << "per DIM, selected by 500-seed survey.\n\n";

        std::cout << "  DIM  -- hypercube dimension; the reservoir has N = 2^DIM neurons\n";
        std::cout << "  HCNN -- HypercubeCNN readout (learned convolution)\n";
        std::cout << "          nl=1 ch=8 FLAT ep=2000 lr=0.0015 bs=1<<(DIM-1)\n\n";

        std::cout << "--- NARMA-10 (NRMSE, lower is better) ---\n";
        std::cout << "Nonlinear autoregressive benchmark requiring both memory (10-step\n";
        std::cout << "history) and nonlinear computation (product terms).\n\n";

        PrintHeader();

        RunDIM<7>(config);
        RunDIM<8>(config);
        RunDIM<9>(config);
        RunDIM<10>(config);
    }

private:
    static void PrintHeader()
    {
        std::cout << "  DIM |     N |      HCNN |  time(s)\n";
        std::cout << "  ----+-------+-----------+---------\n" << std::flush;
    }

    template <size_t DIM>
    static void RunDIM(const ReservoirConfig* config)
    {
        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(4) << (1ULL << DIM) << " |" << std::flush;

        NARMA10<DIM> narma(config);
        auto r = narma.Run();

        std::cout << " " << std::fixed << std::setprecision(6) << std::setw(9) << r.nrmse_hcnn
                  << " | " << std::setprecision(2) << std::setw(7) << r.hcnn_time_s
                  << "\n" << std::flush;
    }
};
