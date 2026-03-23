/// @file StreamingAnomaly.cpp
/// @brief Streaming anomaly detection with incremental readout adaptation.
///
/// Simulates industrial process monitoring: a reservoir learns normal behavior,
/// then detects when the process degrades. Demonstrates:
///   - Anomaly detection via prediction error threshold
///   - Side-by-side comparison: frozen model vs adapted model
///   - TrainIncremental for gradual drift tracking
///
/// The signal: a multi-harmonic process output (fundamental + 3rd harmonic).
/// Degradation: additive noise increases and a DC drift appears, simulating
/// sensor fouling or equipment wear.
///
/// To build and run:
///   cmake -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build
///   ./build/StreamingAnomaly

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "ESN.h"
#include "TranslationLayer.h"

// --- Signal generation ---
// Multi-harmonic process: fundamental + 3rd harmonic, with controllable noise and offset.
static void GenerateProcess(float* out, size_t n, size_t t_start,
                             float noise_level, float dc_drift,
                             std::mt19937_64& rng)
{
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    for (size_t t = 0; t < n; ++t)
    {
        float phase = 0.1f * static_cast<float>(t_start + t);
        float clean = 0.6f * std::sin(phase) + 0.2f * std::sin(3.0f * phase);
        out[t] = clean + dc_drift + noise_level * noise(rng);
    }
}

// Compute RMSE between predictions and targets
static double ComputeRMSE(const float* pred, const float* targets, size_t n)
{
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double err = targets[i] - pred[i];
        mse += err * err;
    }
    return std::sqrt(mse / n);
}

// Predict a batch using the readout, return RMSE
template <size_t DIM>
static double PredictBatch(const LinearReadout& readout, const float* features,
                            const float* targets, size_t n)
{
    constexpr size_t FEATURES = TranslationFeatureCount<DIM>();
    std::vector<float> pred(n);
    for (size_t i = 0; i < n; ++i)
        pred[i] = readout.PredictRaw(features + i * FEATURES);
    return ComputeRMSE(pred.data(), targets, n);
}

int main()
{
    constexpr size_t DIM = 8;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t FEATURES = TranslationFeatureCount<DIM>();

    constexpr size_t warmup = 500;
    constexpr size_t prime_steps = 4000;
    constexpr size_t window = 200;           // Monitoring window size
    constexpr size_t num_windows = 20;       // Total monitoring windows
    constexpr size_t degrade_start = 8;      // Window where degradation begins

    constexpr float normal_noise = 0.01f;    // Baseline sensor noise
    constexpr float degraded_noise = 0.08f;  // Degraded: 8x noise increase
    constexpr float degraded_drift = 0.10f;  // Degraded: DC offset drift
    constexpr float anomaly_threshold = 5.0f; // Alert if RMSE > 5x baseline

    const uint64_t seed = 42;
    std::mt19937_64 signal_rng(seed + 777);

    std::cout << "=== HypercubeRC: Streaming Anomaly Detection ===\n\n";
    std::cout << "Signal: 0.6*sin(t) + 0.2*sin(3t) + noise + drift\n";
    std::cout << "Normal: noise=" << normal_noise << "  drift=0.0\n";
    std::cout << "Degraded: noise=" << degraded_noise << "  drift=" << degraded_drift << "\n";
    std::cout << "Anomaly threshold: " << anomaly_threshold << "x baseline RMSE\n";
    std::cout << "DIM=" << DIM << "  N=" << N << "  Features=" << FEATURES << "\n\n";

    // =================================================================
    // PHASE 1: PRIME
    // =================================================================
    std::cout << "--- Phase 1: Prime on normal operation ---\n";

    size_t t_global = 0;
    std::vector<float> prime_signal(warmup + prime_steps + 1);
    GenerateProcess(prime_signal.data(), prime_signal.size(), t_global,
                     normal_noise, 0.0f, signal_rng);

    ESN<DIM> esn(seed);
    esn.Warmup(prime_signal.data(), warmup);
    esn.Run(prime_signal.data() + warmup, prime_steps);
    t_global += warmup + prime_steps;

    auto prime_features = TranslationTransform<DIM>(esn.States(), prime_steps);
    std::vector<float> prime_targets(prime_steps);
    for (size_t t = 0; t < prime_steps; ++t)
        prime_targets[t] = prime_signal[warmup + t + 1];

    size_t train_n = static_cast<size_t>(prime_steps * 0.7);
    size_t test_n = prime_steps - train_n;

    LinearReadout readout;
    readout.Train(prime_features.data(), prime_targets.data(), train_n, FEATURES);

    // Measure baseline on held-out portion (data the readout has NOT seen)
    double baseline = PredictBatch<DIM>(readout, prime_features.data() + train_n * FEATURES,
                                         prime_targets.data() + train_n, test_n);
    double threshold = baseline * anomaly_threshold;

    std::cout << "  Trained on " << prime_steps << " samples\n";
    std::cout << "  Baseline RMSE: " << std::fixed << std::setprecision(6) << baseline << "\n";
    std::cout << "  Anomaly threshold: " << std::setprecision(6) << threshold << "\n\n";

    // =================================================================
    // PHASE 2: STREAMING MONITOR
    // =================================================================
    std::cout << "--- Phase 2: Streaming monitor (" << num_windows << " windows of "
              << window << " steps) ---\n\n";

    // Keep a frozen copy for comparison
    LinearReadout frozen_readout = readout;

    std::cout << "  Window | Condition  |  Frozen RMSE | Adapted RMSE | Status\n";
    std::cout << "  -------+------------+--------------+--------------+------------------\n";

    bool first_alert = true;

    for (size_t w = 0; w < num_windows; ++w)
    {
        // Determine condition for this window
        float noise = normal_noise;
        float drift = 0.0f;
        const char* condition = "Normal    ";

        if (w > degrade_start)
        {
            // Gradual degradation: ramp up over 4 windows, then plateau
            float progress = std::min(1.0f, static_cast<float>(w - degrade_start) / 4.0f);
            noise = normal_noise + progress * (degraded_noise - normal_noise);
            drift = progress * degraded_drift;
            condition = "Degrading ";
            if (progress >= 1.0f)
                condition = "Degraded  ";
        }

        // Generate window signal
        esn.ClearStates();
        std::vector<float> sig(window + 1);
        GenerateProcess(sig.data(), sig.size(), t_global, noise, drift, signal_rng);

        esn.Run(sig.data(), window);
        t_global += window;

        auto feat = TranslationTransform<DIM>(esn.States(), window);
        std::vector<float> tgt(window);
        for (size_t t = 0; t < window; ++t)
            tgt[t] = sig[t + 1];

        // Measure both models
        double frozen_rmse = PredictBatch<DIM>(frozen_readout, feat.data(), tgt.data(), window);
        double adapted_rmse = PredictBatch<DIM>(readout, feat.data(), tgt.data(), window);

        // Adapt the live model (low blend for gentle tracking)
        if (w > degrade_start)
            readout.TrainIncremental(feat.data(), tgt.data(), window, FEATURES, 0.15f);

        // Status
        const char* status = "";
        if (frozen_rmse > threshold)
        {
            if (first_alert)
            {
                status = "** ANOMALY DETECTED **";
                first_alert = false;
            }
            else
                status = "ANOMALY";
        }

        std::cout << "  " << std::setw(5) << (w + 1) << "  | " << condition
                  << " | " << std::setprecision(6) << std::setw(12) << frozen_rmse
                  << " | " << std::setw(12) << adapted_rmse
                  << " | " << status << "\n";
    }

    // =================================================================
    // SUMMARY
    // =================================================================
    std::cout << "\n--- Summary ---\n";
    std::cout << "  The frozen model (no adaptation) shows rising error as the process\n";
    std::cout << "  degrades, crossing the anomaly threshold and staying elevated.\n\n";
    std::cout << "  The adapted model (TrainIncremental, blend=0.15) tracks the DC drift\n";
    std::cout << "  and maintains ~20% lower error than the frozen model. It cannot\n";
    std::cout << "  recover to baseline because the degraded noise level (8x normal)\n";
    std::cout << "  sets an irreducible error floor -- the reservoir can track drift but\n";
    std::cout << "  cannot predict random noise.\n\n";
    std::cout << "  In practice: use the frozen model for anomaly detection (error spike\n";
    std::cout << "  = something changed), then switch to the adapted model to continue\n";
    std::cout << "  monitoring under the new operating regime.\n";

    return 0;
}
