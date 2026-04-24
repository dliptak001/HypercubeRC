#pragma once

#include <iostream>
#include <iomanip>
#include "NARMA10.h"

/// Unified NARMA-10 benchmark suite across DIM 7-10.
/// Runs each DIM at depth 1 (single reservoir) and depth 2 (cascade).
struct BenchmarkSuite
{
    static constexpr int   kNumLayers    = 1;
    static constexpr int   kConvChannels = 8;
    static constexpr int   kEpochs       = 1000;
    static constexpr float kLrMax        = 0.0015f;

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
        std::cout << "          nl=" << kNumLayers << " ch=" << kConvChannels
                  << " ep=" << kEpochs << " lr=" << kLrMax
                  << " bs=1<<(DIM-1)\n\n";

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
        std::cout << "  DIM |     N | d1 NRMSE   | d2 NRMSE   |   diff\n";
        std::cout << "  ----+-------+------------+------------+--------\n" << std::flush;
    }

    template <size_t DIM>
    static void RunDIM(const ReservoirConfig* config)
    {
        constexpr size_t N = 1ULL << DIM;

        ReadoutArchConfig arch;
        arch.num_layers    = kNumLayers;
        arch.conv_channels = kConvChannels;
        arch.seed          = presets::Baseline<DIM>().arch.seed;

        ReadoutTrainConfig train;
        train.epochs     = kEpochs;
        train.batch_size = 1 << (DIM - 1);
        train.lr_max     = kLrMax;

        std::cout << "  " << std::setw(3) << DIM
                  << "  | " << std::setw(4) << N << " |" << std::flush;

        NARMA10<DIM> narma1(config, 1, arch, train);
        auto r1 = narma1.Run();

        std::cout << " " << std::fixed << std::setprecision(6) << std::setw(10) << r1.nrmse_hcnn
                  << " |" << std::flush;

        NARMA10<DIM> narma2(config, 2, arch, train);
        auto r2 = narma2.Run();

        double pct = (r2.nrmse_hcnn - r1.nrmse_hcnn) / r1.nrmse_hcnn * 100.0;
        std::cout << " " << std::fixed << std::setprecision(6) << std::setw(10) << r2.nrmse_hcnn
                  << " | " << std::showpos << std::setprecision(1) << std::setw(5) << pct << "%"
                  << std::noshowpos << "\n" << std::flush;
    }
};
