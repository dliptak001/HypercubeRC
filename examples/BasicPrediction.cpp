/// @file BasicPrediction.cpp
/// @brief Sine prediction demo: single reservoir vs. cascade comparison.
/// See BasicPrediction.md for walkthrough and experiments.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "ESN.h"
#include "Presets.h"

struct RunResult
{
    double r2;
    double nrmse;
    double train_secs;
};

template <size_t DIM>
RunResult run_prediction(size_t depth, const char* label)
{
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t warmup = 200;
    constexpr size_t collect = 2000;
    constexpr size_t horizon = 1;
    constexpr double train_fraction = 0.7;
    const uint64_t seed = SurveyedSeed<DIM>();

    const size_t total = warmup + collect + horizon;
    std::vector<float> signal(total);
    for (size_t t = 0; t < total; ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + horizon];

    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;

    ReservoirConfig cfg;
    cfg.seed = seed;
    cfg.leak_rate = 0.2f;
    cfg.output_fraction = 1.0f;
    ESN<DIM> esn(depth, cfg);

    auto preset = presets::Baseline<DIM>();
    ReadoutArchConfig cnn_arch = preset.arch;
    ReadoutTrainConfig cnn_train = preset.train;
    cnn_train.epochs      = 1100;
    cnn_train.lr_min_frac = 0.1f;
    cnn_arch.seed         = 42007;

    std::cout << "--- " << label << " ---\n";
    std::cout << "  DIM=" << DIM << "  N=" << N << "  depth=" << depth
              << "  neurons=" << (depth * N)
              << "  output_size=" << esn.OutputSize() << "\n";
    std::cout << "  Training: " << cnn_train.epochs << " epochs, batch=" << cnn_train.batch_size
              << ", lr=" << cnn_train.lr_max
              << " (cosine, floor=" << (cnn_train.lr_max * cnn_train.lr_min_frac) << ")\n";

    esn.Warmup(signal.data(), warmup);
    esn.Run(signal.data() + warmup, collect);

    std::cout << "  Training..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(targets.data(), train_size, cnn_arch, cnn_train);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n";

    double r2 = esn.R2(targets.data(), train_size, test_size);
    double nrmse = esn.NRMSE(targets.data(), train_size, test_size);

    std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2;
    if (r2 > 0.9999) std::cout << "  (effectively perfect)";
    else if (r2 > 0.99) std::cout << "  (excellent)";
    else if (r2 > 0.9) std::cout << "  (good)";
    std::cout << "\n";
    std::cout << "  NRMSE: " << std::setprecision(6) << nrmse;
    if (nrmse < 0.001) std::cout << "  (sub-0.1% error)";
    else if (nrmse < 0.01) std::cout << "  (under 1% error)";
    else if (nrmse < 0.1) std::cout << "  (under 10% error)";
    std::cout << "\n";

    std::cout << "\n  Sample predictions (test set):\n";
    std::cout << "    Step  |   Actual   |  Predicted  |    Error\n";
    std::cout << "    ------+------------+-------------+-----------\n";
    for (size_t i = 0; i < 10; ++i)
    {
        float actual = targets[train_size + i];
        float predicted = esn.PredictRaw(train_size + i);
        float error = actual - predicted;
        std::cout << "    " << std::setw(5) << (train_size + i)
                  << " | " << std::showpos << std::setprecision(5) << std::setw(10) << actual
                  << " | " << std::setw(11) << predicted
                  << " | " << std::setw(10) << error
                  << std::noshowpos << "\n";
    }
    std::cout << "\n";

    return {r2, nrmse, secs};
}

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    std::cout << "=== HypercubeRC: Sine Wave Prediction ===\n\n";
    std::cout << "Task: predict the next value of sin(0.1t) from the reservoir's\n";
    std::cout << "internal state.  The readout never sees the input directly -- it\n";
    std::cout << "learns the input-to-output mapping entirely from reservoir dynamics.\n\n";

    auto r1 = run_prediction<7>(1, "Single Reservoir (DIM 7, depth 1)");
    auto r2 = run_prediction<6>(2, "Cascade (DIM 6, depth 2)");

    std::cout << "=== Comparison ===\n\n";
    std::cout << "                        |  DIM 7 x 1  |  DIM 6 x 2\n";
    std::cout << "  ----------------------+-------------+-------------\n";
    std::cout << "  Neurons               |     "
              << std::setw(5) << (1 << 7) << "    |     "
              << std::setw(5) << (2 * (1 << 6)) << "\n";
    std::cout << "  Output features       |     "
              << std::setw(5) << (1 << 7) << "    |     "
              << std::setw(5) << (2 * (1 << 6)) << "\n";
    std::cout << std::fixed;
    std::cout << "  R2                    |  " << std::setprecision(6) << std::setw(11) << r1.r2
              << "  |  " << std::setw(11) << r2.r2 << "\n";
    std::cout << "  NRMSE                 |  " << std::setprecision(6) << std::setw(11) << r1.nrmse
              << "  |  " << std::setw(11) << r2.nrmse << "\n";
    std::cout << "  Train time (s)        |  " << std::setprecision(1) << std::setw(11) << r1.train_secs
              << "  |  " << std::setw(11) << r2.train_secs << "\n";
    std::cout << "Note: sine prediction is a smooth, short-memory signal -- a single\n";
    std::cout << "reservoir is already effectively perfect. The cascade's memory\n";
    std::cout << "extension has no headroom to exploit here. Long-memory tasks\n";
    std::cout << "(NARMA-10, char-level LM) are the intended cascade benchmark.\n\n";

    return 0;
}
