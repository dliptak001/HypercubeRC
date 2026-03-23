/// @file BasicPrediction.cpp
/// @brief Minimal example: predict a sine wave using HypercubeRC.
///
/// This is the simplest possible end-to-end demo of the reservoir computing
/// pipeline. It generates a sine wave, feeds it into the reservoir, trains
/// a linear readout to predict the next value, and reports the error.
///
/// The pipeline:
///   1. Generate input signal (sine wave)
///   2. Create ESN and drive it (warmup + collect)
///   3. Apply translation layer (N -> 2.5N features)
///   4. Train LinearReadout on translated features
///   5. Evaluate prediction quality on held-out test set
///
/// To build and run:
///   cmake -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build
///   ./build/BasicPrediction

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "ESN.h"
#include "TranslationLayer.h"

int main()
{
    // --- Configuration ---
    constexpr size_t DIM = 7;                // Hypercube dimension (2^7 = 128 neurons)
    constexpr size_t N = 1ULL << DIM;        // Reservoir size
    constexpr size_t FEATURES = TranslationFeatureCount<DIM>();  // 2.5N = 320

    constexpr size_t warmup = 200;           // Steps to wash out initial transients
    constexpr size_t collect = 2000;         // Steps to collect training + test data
    constexpr size_t horizon = 1;            // Predict 1 step ahead
    constexpr double train_fraction = 0.7;   // 70% train, 30% test

    const uint64_t seed = 42;               // Reservoir random seed (deterministic)

    std::cout << "=== HypercubeRC: Basic Sine Wave Prediction ===\n\n";
    std::cout << "DIM=" << DIM << "  N=" << N << "  Features=" << FEATURES << "\n";
    std::cout << "Warmup=" << warmup << "  Collect=" << collect
              << "  Horizon=" << horizon << "\n\n";

    // --- Step 1: Generate a sine wave input signal ---
    // The reservoir will learn to predict the next value of this signal.
    // In a real application, this would be your sensor data, time series, etc.
    // Note: all inputs must be in [-1, 1] — values outside this range are silently clamped.
    const size_t total = warmup + collect + horizon;
    std::vector<float> signal(total);
    for (size_t t = 0; t < total; ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));  // Amplitude 1.0, stays in [-1, 1]

    // --- Step 2: Create ESN and drive the reservoir ---
    ESN<DIM> esn(seed);

    // Warmup: drive the reservoir without recording, to wash out initial conditions.
    // After warmup, the reservoir state reflects the input history.
    esn.Warmup(signal.data(), warmup);

    // Run: drive and record the N-dimensional state at each timestep.
    esn.Run(signal.data() + warmup, collect);

    std::cout << "Reservoir driven: " << collect << " states collected.\n";

    // --- Step 3: Apply translation layer ---
    // Expand N raw states into 2.5N features (x, x², x*x') for the readout.
    auto features = TranslationTransform<DIM>(esn.States(), collect);

    std::cout << "Translation applied: " << N << " -> " << FEATURES << " features per step.\n";

    // --- Step 4: Build targets and train the readout ---
    // Target: the signal value `horizon` steps into the future.
    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + horizon];

    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;

    LinearReadout readout;
    readout.Train(features.data(), targets.data(), train_size, FEATURES);

    std::cout << "Readout trained on " << train_size << " samples.\n\n";

    // --- Step 5: Evaluate on the test set ---
    const float* test_features = features.data() + train_size * FEATURES;
    const float* test_targets = targets.data() + train_size;

    double r2 = readout.R2(test_features, test_targets, test_size);

    // Compute NRMSE manually for display
    double mean = 0.0;
    for (size_t i = 0; i < test_size; ++i) mean += test_targets[i];
    mean /= test_size;

    double var = 0.0, mse = 0.0;
    for (size_t i = 0; i < test_size; ++i)
    {
        double y = test_targets[i];
        double yh = readout.PredictRaw(test_features + i * FEATURES);
        var += (y - mean) * (y - mean);
        mse += (y - yh) * (y - yh);
    }
    double nrmse = std::sqrt(mse / test_size) / std::sqrt(var / test_size);

    std::cout << "--- Results (test set: " << test_size << " samples) ---\n";
    std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2 << "\n";
    std::cout << "  NRMSE: " << std::setprecision(6) << nrmse << "\n\n";

    // --- Show a few predictions vs actual ---
    std::cout << "--- Sample predictions ---\n";
    std::cout << "  Step  |  Actual  | Predicted |   Error\n";
    std::cout << "  ------+----------+-----------+---------\n";
    for (size_t i = 0; i < 10; ++i)
    {
        float actual = test_targets[i];
        float predicted = readout.PredictRaw(test_features + i * FEATURES);
        float error = actual - predicted;
        std::cout << "  " << std::setw(5) << (train_size + i)
                  << " | " << std::showpos << std::setprecision(4) << std::setw(8) << actual
                  << " | " << std::setw(9) << predicted
                  << " | " << std::setw(8) << error
                  << std::noshowpos << "\n";
    }

    std::cout << "\nDone. The reservoir learned to predict sin(t+1) from sin(t).\n";
    return 0;
}
