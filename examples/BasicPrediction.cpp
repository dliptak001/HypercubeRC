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
///   3. Optionally apply translation layer (N -> 2.5N features)
///   4. Train LinearReadout on features
///   5. Evaluate prediction quality on held-out test set
///
/// To build and run:
///   cmake -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build
///   ./build/BasicPrediction            (default: translation features)
///   ./build/BasicPrediction raw        (raw N features)
///   ./build/BasicPrediction translation (explicit translation)

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include "ESN.h"
#include "Reservoir.h"
#include "TranslationLayer.h"
#include "readout/RidgeRegression.h"

int main(int argc, char* argv[])
{
    // --- Parse feature mode ---
    bool use_translation = false;  // default
    if (argc > 1)
    {
        if (std::strcmp(argv[1], "raw") == 0)
            use_translation = false;
        else if (std::strcmp(argv[1], "translation") == 0)
            use_translation = true;
        else
        {
            std::cerr << "Usage: " << argv[0] << " [raw|translation]\n";
            return 1;
        }
    }

    // --- Configuration ---
    constexpr size_t DIM = 7;                // Hypercube dimension (2^7 = 128 neurons)
    constexpr size_t N = 1ULL << DIM;        // Reservoir size
    constexpr size_t FEATURES_TRANS = TranslationFeatureCount<DIM>();  // 2.5N = 320

    constexpr size_t warmup = 200;           // Steps to wash out initial transients
    constexpr size_t collect = 2000;         // Steps to collect training + test data
    constexpr size_t horizon = 1;            // Predict 1 step ahead
    constexpr double train_fraction = 0.7;   // 70% train, 30% test

    constexpr uint64_t seed = 42;            // Reservoir random seed (deterministic)
    const size_t num_features = use_translation ? FEATURES_TRANS : N;

    std::cout << "=== HypercubeRC: Basic Sine Wave Prediction ===\n\n";
    std::cout << "Task: predict the next value of sin(0.1t) from the reservoir's\n";
    std::cout << "internal state. The readout never sees the input directly -- it\n";
    std::cout << "learns the input-to-output mapping entirely from reservoir dynamics.\n\n";

    // --- Step 1: Generate a sine wave input signal ---
    // The reservoir will learn to predict the next value of this signal.
    // In a real application, this would be your sensor data, time series, etc.
    // Note: all inputs must be in [-1, 1] — values outside this range are silently clamped.
    const size_t total = warmup + collect + horizon;
    std::vector<float> signal(total);
    for (size_t t = 0; t < total; ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));  // Amplitude 1.0, stays in [-1, 1]

    // --- Step 2: Create ESN and drive the reservoir ---
    auto mode = use_translation ? FeatureMode::Translation : FeatureMode::Raw;
    ESN<DIM> esn(seed, ReadoutType::Ridge, mode);

    const char* readout_label = (esn.GetReadoutType() == ReadoutType::Ridge) ? "Ridge" : "Linear";
    std::cout << "Config: DIM=" << DIM << "  N=" << N << "  Features=" << num_features
              << " (" << (use_translation ? "translation" : "raw") << ")"
              << "  Readout=" << readout_label
              << "  Horizon=" << horizon << "\n\n";

    // Warmup: drive the reservoir without recording, to wash out initial conditions.
    // After warmup, the reservoir state reflects the input history.
    esn.Warmup(signal.data(), warmup);

    // Run: drive and record the N-dimensional state at each timestep.
    esn.Run(signal.data() + warmup, collect);

    // --- Step 3: Get features (raw or translated) ---
    const float* features = nullptr;
    std::vector<float> translated;
    if (use_translation)
    {
        translated = TranslationTransform<DIM>(esn.States(), collect);
        features = translated.data();
    }
    else
    {
        features = esn.States();
    }

    // --- Step 4: Build targets and train the readout ---
    // Target: the signal value `horizon` steps into the future.
    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + horizon];

    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;

    // Train readout and evaluate — generic over readout type.
    const float* test_features = features + train_size * num_features;
    const float* test_targets = targets.data() + train_size;

    auto train_and_evaluate = [&](auto& readout)
    {
        readout.Train(features, targets.data(), train_size, num_features);

        std::cout << "--- Pipeline ---\n";
        std::cout << "  1. Generated " << (warmup + collect) << " samples of sin(0.1t)\n";
        std::cout << "  2. Drove reservoir for " << warmup << " warmup + "
                  << collect << " recorded steps\n";
        std::cout << "  3. Extracted " << num_features << " features per step"
                  << (use_translation ? " (x + x^2 + x*x' translation)" : " (raw states)")
                  << "\n";
        std::cout << "  4. Trained " << readout_label << " readout on " << train_size
                  << " samples, testing on " << test_size << "\n\n";

        // --- Step 5: Evaluate on the test set ---
        double r2 = readout.R2(test_features, test_targets, test_size);

        // Compute NRMSE manually for display
        double mean = 0.0;
        for (size_t i = 0; i < test_size; ++i) mean += test_targets[i];
        mean /= test_size;

        double var = 0.0, mse = 0.0;
        for (size_t i = 0; i < test_size; ++i)
        {
            double y = test_targets[i];
            double yh = readout.PredictRaw(test_features + i * num_features);
            var += (y - mean) * (y - mean);
            mse += (y - yh) * (y - yh);
        }
        double nrmse = std::sqrt(mse / test_size) / std::sqrt(var / test_size);

        std::cout << "--- How well does it predict? ---\n\n";
        std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2;
        if (r2 > 0.9999)
            std::cout << "  (effectively perfect)";
        else if (r2 > 0.99)
            std::cout << "  (excellent)";
        std::cout << "\n";
        std::cout << "  NRMSE: " << std::setprecision(6) << nrmse;
        if (nrmse < 0.001)
            std::cout << "  (sub-0.1% error)";
        else if (nrmse < 0.01)
            std::cout << "  (under 1% error)";
        std::cout << "\n\n";

        // --- Show a few predictions vs actual ---
        std::cout << "Sample predictions (test set, never seen during training):\n\n";
        std::cout << "  Step  |  Actual  | Predicted |   Error\n";
        std::cout << "  ------+----------+-----------+---------\n";
        for (size_t i = 0; i < 10; ++i)
        {
            float actual = test_targets[i];
            float predicted = readout.PredictRaw(test_features + i * num_features);
            float error = actual - predicted;
            std::cout << "  " << std::setw(5) << (train_size + i)
                      << " | " << std::showpos << std::setprecision(4) << std::setw(8) << actual
                      << " | " << std::setw(9) << predicted
                      << " | " << std::setw(8) << error
                      << std::noshowpos << "\n";
        }

        std::cout << "\nThe readout learned sin(t+1) from " << num_features
                  << " reservoir state features.\n";
        std::cout << "Errors are in the 4th decimal place -- the reservoir's nonlinear\n";
        std::cout << "dynamics encode enough of the input history for near-exact prediction.\n";
    };

    if (esn.GetReadoutType() == ReadoutType::Ridge)
    {
        RidgeRegression readout;
        train_and_evaluate(readout);
    }
    else
    {
        LinearReadout readout;
        train_and_evaluate(readout);
    }
    return 0;
}
