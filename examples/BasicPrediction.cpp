/// @file BasicPrediction.cpp
/// @brief Predict a sine wave using HypercubeRC with Ridge and HCNN readouts.
///
/// The simplest end-to-end reservoir computing demo.  A sine wave is fed into
/// the reservoir, and two readouts learn to predict the next value from the
/// reservoir's internal state alone:
///   - Ridge regression on stride-selected features (fast, closed-form)
///   - HypercubeCNN on raw state (learned convolutional readout)
///
/// Both readouts are trained and evaluated on identical data from the same
/// reservoir run, giving an apples-to-apples comparison.
///
/// See BasicPrediction.md for a detailed walkthrough and suggested experiments.

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include "ESN.h"
#include "readout/HCNNPresets.h"

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    // --- Configuration ---
    constexpr size_t DIM = 7;                // Hypercube dimension (2^7 = 128 neurons)
    constexpr size_t N = 1ULL << DIM;        // Reservoir size
    constexpr size_t warmup = 200;           // Steps to wash out initial transients
    constexpr size_t collect = 2000;         // Steps to collect training + test data
    constexpr size_t horizon = 1;            // Predict 1 step ahead
    constexpr double train_fraction = 0.7;   // 70% train, 30% test

    constexpr uint64_t seed = 6437149480297576047ULL;  // NARMA-10 best seed for DIM 7

    std::cout << "=== HypercubeRC: Sine Wave Prediction ===\n\n";
    std::cout << "Task: predict the next value of sin(0.1t) from the reservoir's\n";
    std::cout << "internal state.  The readout never sees the input directly -- it\n";
    std::cout << "learns the input-to-output mapping entirely from reservoir dynamics.\n\n";

    // --- Step 1: Generate a sine wave input signal ---
    const size_t total = warmup + collect + horizon;
    std::vector<float> signal(total);
    for (size_t t = 0; t < total; ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    // --- Step 2: Build targets ---
    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + horizon];

    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;

    // --- Common reservoir config ---
    ReservoirConfig cfg;
    cfg.seed = seed;
    cfg.leak_rate = 0.2f;

    // ===================================================================
    //  Ridge readout
    // ===================================================================
    std::cout << "--- Ridge readout ---\n\n";

    ReservoirConfig ridge_cfg = cfg;
    ridge_cfg.output_fraction = 0.1f;
    ESN<DIM> esn_ridge(ridge_cfg, ReadoutType::Ridge);

    std::cout << "  Config: N=" << N
              << "  Outputs=" << esn_ridge.NumOutputVerts()
              << " (" << static_cast<int>(esn_ridge.OutputFraction() * 100) << "%)"
              << "  Features=" << esn_ridge.NumFeatures() << "\n";

    esn_ridge.Warmup(signal.data(), warmup);
    esn_ridge.Run(signal.data() + warmup, collect);
    esn_ridge.Train(targets.data(), train_size);

    double r2_ridge = esn_ridge.R2(targets.data(), train_size, test_size);
    double nrmse_ridge = esn_ridge.NRMSE(targets.data(), train_size, test_size);

    std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2_ridge;
    if (r2_ridge > 0.9999) std::cout << "  (effectively perfect)";
    else if (r2_ridge > 0.99) std::cout << "  (excellent)";
    std::cout << "\n";
    std::cout << "  NRMSE: " << std::setprecision(6) << nrmse_ridge;
    if (nrmse_ridge < 0.001) std::cout << "  (sub-0.1% error)";
    else if (nrmse_ridge < 0.01) std::cout << "  (under 1% error)";
    std::cout << "\n\n";

    // ===================================================================
    //  HCNN readout
    // ===================================================================
    std::cout << "--- HCNN readout ---\n\n";

    ReservoirConfig hcnn_cfg_r = cfg;
    hcnn_cfg_r.output_fraction = 1.0f;  // CNN uses all vertices
    ESN<DIM> esn_hcnn(hcnn_cfg_r, ReadoutType::HCNN);

    // HRCCNN baseline architecture (nl=1, ch=8, FLAT, lr=0.0015,
    // bs=1<<(DIM-1)) with smooth-signal epochs: ep=100 stays because
    // BasicPrediction is a sine wave (not chaotic), and the baseline's
    // ep=2000 is calibrated for chaotic signals (NARMA).
    CNNReadoutConfig cnn_cfg = hcnn_presets::HRCCNNBaseline<DIM>();
    cnn_cfg.epochs      = 1100;
    cnn_cfg.lr_min_frac = 0.1f;
    cnn_cfg.seed        = 42007;

    std::cout << "  Config: N=" << N << "  raw state (all vertices)\n";
    std::cout << "  Training: " << cnn_cfg.epochs << " epochs, batch=" << cnn_cfg.batch_size
              << ", lr=" << cnn_cfg.lr_max
              << " (cosine, floor=" << (cnn_cfg.lr_max * cnn_cfg.lr_min_frac) << ")\n";

    esn_hcnn.Warmup(signal.data(), warmup);
    esn_hcnn.Run(signal.data() + warmup, collect);

    std::cout << "  Training..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn_hcnn.Train(targets.data(), train_size, cnn_cfg);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n";

    double r2_hcnn = esn_hcnn.R2(targets.data(), train_size, test_size);
    double nrmse_hcnn = esn_hcnn.NRMSE(targets.data(), train_size, test_size);

    std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2_hcnn;
    if (r2_hcnn > 0.9999) std::cout << "  (effectively perfect)";
    else if (r2_hcnn > 0.99) std::cout << "  (excellent)";
    else if (r2_hcnn > 0.9) std::cout << "  (good)";
    std::cout << "\n";
    std::cout << "  NRMSE: " << std::setprecision(6) << nrmse_hcnn;
    if (nrmse_hcnn < 0.001) std::cout << "  (sub-0.1% error)";
    else if (nrmse_hcnn < 0.01) std::cout << "  (under 1% error)";
    else if (nrmse_hcnn < 0.1) std::cout << "  (under 10% error)";
    std::cout << "\n\n";

    // ===================================================================
    //  Sample predictions (HCNN)
    // ===================================================================
    std::cout << "Sample predictions (HCNN, test set):\n\n";
    std::cout << "  Step  |   Actual   |  Predicted  |    Error\n";
    std::cout << "  ------+------------+-------------+-----------\n";
    for (size_t i = 0; i < 10; ++i)
    {
        float actual = targets[train_size + i];
        float predicted = esn_hcnn.PredictRaw(train_size + i);
        float error = actual - predicted;
        std::cout << "  " << std::setw(5) << (train_size + i)
                  << " | " << std::showpos << std::setprecision(5) << std::setw(10) << actual
                  << " | " << std::setw(11) << predicted
                  << " | " << std::setw(10) << error
                  << std::noshowpos << "\n";
    }

    // ===================================================================
    //  Comparison
    // ===================================================================
    std::cout << "\n--- Comparison ---\n\n";
    std::cout << "  Readout |  R2         |  NRMSE\n";
    std::cout << "  --------+-------------+-----------\n";
    std::cout << "  Ridge   | " << std::setprecision(6) << std::setw(11) << r2_ridge
              << " | " << std::setw(10) << nrmse_ridge << "\n";
    std::cout << "  HCNN    | " << std::setw(11) << r2_hcnn
              << " | " << std::setw(10) << nrmse_hcnn << "\n";

    std::cout << "\nBoth readouts learned sin(t+1) from the same reservoir dynamics.\n";
    std::cout << "Ridge uses hand-crafted features; HCNN discovers features via convolution.\n";

    return 0;
}
