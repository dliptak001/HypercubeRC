/// @file CNNPrediction.cpp
/// @brief Predict a sine wave using HypercubeRC with HypercubeCNN as the readout.
///
/// Mirrors BasicPrediction.cpp but replaces the Ridge readout with a learned
/// CNN readout.  The reservoir's raw state (N = 2^DIM floats per timestep) is
/// fed directly into HypercubeCNN -- no stride selection, no translation layer,
/// no hand-crafted features.  The CNN's convolution kernels discover which
/// vertex interactions predict the target.
///
/// This is the end-to-end validation for the HCNN-as-RC-readout integration.
/// Compare R^2 and NRMSE against BasicPrediction's Ridge baseline.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "ESN.h"

int main()
{
    constexpr size_t DIM = 7;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t warmup = 200;
    constexpr size_t collect = 2000;
    constexpr size_t horizon = 1;
    constexpr double train_fraction = 0.7;

    constexpr uint64_t seed = 6437149480297576047ULL;  // same seed as BasicPrediction

    std::cout << "=== HypercubeRC + HypercubeCNN: CNN Readout Prediction ===\n\n";
    std::cout << "Task: predict sin(0.1*(t+1)) from the reservoir's raw state.\n";
    std::cout << "Readout: HypercubeCNN (Conv -> MaxPool -> GAP -> Linear).\n";
    std::cout << "No translation layer, no crafted features -- raw state only.\n\n";

    // --- Generate sine wave ---
    const size_t total = warmup + collect + horizon;
    std::vector<float> signal(total);
    for (size_t t = 0; t < total; ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    // --- Create ESN with CNN readout ---
    ReservoirConfig cfg;
    cfg.seed = seed;
    cfg.leak_rate = 0.2f;
    // CNN uses ALL vertices (no stride selection).
    cfg.output_fraction = 1.0f;

    ESN<DIM> esn(cfg, ReadoutType::HCNN);

    std::cout << "Reservoir: DIM=" << DIM << "  N=" << N
              << "  leak_rate=" << cfg.leak_rate << "\n";

    // --- Drive reservoir ---
    esn.Warmup(signal.data(), warmup);
    esn.Run(signal.data() + warmup, collect);

    // --- Build targets ---
    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + horizon];

    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;

    // --- Configure CNN readout ---
    CNNReadoutConfig cnn_cfg;
    cnn_cfg.conv_channels = 16;
    cnn_cfg.epochs = 100;
    cnn_cfg.batch_size = 32;
    cnn_cfg.lr_max = 0.005f;
    cnn_cfg.lr_min_frac = 0.1f;
    cnn_cfg.seed = 42;

    // Auto-sized architecture: min(DIM - 3, 4) Conv+Pool pairs, channels doubling.
    const int num_layers = std::min(static_cast<int>(DIM) - 3, 4);
    int total_params = 0;
    int ch_in = 1;
    int layer_dim = static_cast<int>(DIM);
    std::cout << "CNN:       ";
    for (int i = 0; i < num_layers; ++i) {
        int ch_out = cnn_cfg.conv_channels * (1 << i);
        int layer_params = ch_in * ch_out * layer_dim + ch_out;  // kernel + bias
        total_params += layer_params;
        if (i > 0) std::cout << " -> ";
        std::cout << "Conv(" << ch_out << ",K=" << layer_dim << ") -> Pool";
        ch_in = ch_out;
        layer_dim -= 1;
    }
    int readout_params = ch_in + 1;  // GAP reduces to ch_in features -> Linear(1)
    total_params += readout_params;
    std::cout << " -> GAP -> Linear(1)\n";
    std::cout << "           " << total_params << " parameters ("
              << num_layers << " conv+pool layers)\n";
    std::cout << "Training:  " << cnn_cfg.epochs << " epochs, batch=" << cnn_cfg.batch_size
              << ", lr=" << cnn_cfg.lr_max << " (cosine, floor=" << (cnn_cfg.lr_max * cnn_cfg.lr_min_frac) << ")\n";
    std::cout << "Data:      " << train_size << " train, " << test_size << " test\n\n";

    // --- Train ---
    std::cout << "Training CNN readout..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(targets.data(), train_size, cnn_cfg);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n\n";

    // --- Evaluate ---
    double r2 = esn.R2(targets.data(), train_size, test_size);
    double nrmse = esn.NRMSE(targets.data(), train_size, test_size);

    std::cout << "--- Results ---\n\n";
    std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2;
    if (r2 > 0.9999)
        std::cout << "  (effectively perfect)";
    else if (r2 > 0.99)
        std::cout << "  (excellent)";
    else if (r2 > 0.9)
        std::cout << "  (good)";
    std::cout << "\n";
    std::cout << "  NRMSE: " << std::setprecision(6) << nrmse;
    if (nrmse < 0.001)
        std::cout << "  (sub-0.1% error)";
    else if (nrmse < 0.01)
        std::cout << "  (under 1% error)";
    else if (nrmse < 0.1)
        std::cout << "  (under 10% error)";
    std::cout << "\n\n";

    // --- Sample predictions ---
    std::cout << "Sample predictions (test set):\n\n";
    std::cout << "  Step  |   Actual   |  Predicted  |    Error\n";
    std::cout << "  ------+------------+-------------+-----------\n";
    for (size_t i = 0; i < 10; ++i) {
        float actual = targets[train_size + i];
        float predicted = esn.PredictRaw(train_size + i);
        float error = actual - predicted;
        std::cout << "  " << std::setw(5) << (train_size + i)
                  << " | " << std::showpos << std::setprecision(5) << std::setw(10) << actual
                  << " | " << std::setw(11) << predicted
                  << " | " << std::setw(10) << error
                  << std::noshowpos << "\n";
    }

    // --- Ridge baseline for comparison ---
    std::cout << "\n--- Ridge baseline (same reservoir, same data) ---\n\n";
    ESN<DIM> esn_ridge(cfg, ReadoutType::Ridge, FeatureMode::Raw);
    esn_ridge.Warmup(signal.data(), warmup);
    esn_ridge.Run(signal.data() + warmup, collect);
    esn_ridge.Train(targets.data(), train_size);
    double r2_ridge = esn_ridge.R2(targets.data(), train_size, test_size);
    double nrmse_ridge = esn_ridge.NRMSE(targets.data(), train_size, test_size);
    std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2_ridge << "\n";
    std::cout << "  NRMSE: " << std::setprecision(6) << nrmse_ridge << "\n\n";

    std::cout << "--- Comparison ---\n\n";
    std::cout << "  Readout  |  R2        |  NRMSE     |  Parameters\n";
    std::cout << "  ---------+------------+------------+------------\n";
    std::cout << "  CNN      | " << std::setprecision(6) << std::setw(10) << r2
              << " | " << std::setw(10) << nrmse
              << " | " << std::setw(10) << total_params << "\n";
    std::cout << "  Ridge    | " << std::setw(10) << r2_ridge
              << " | " << std::setw(10) << nrmse_ridge
              << " | " << std::setw(10) << esn_ridge.NumFeatures() << "\n";

    return 0;
}
