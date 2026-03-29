/// @file StreamingAnomaly.cpp
/// @brief Streaming anomaly detection example.
///
/// Simulates industrial process monitoring: a reservoir learns normal behavior,
/// then detects anomalies when the process deviates. Three distinct anomaly
/// events are injected — a noise spike, a DC drift, and a frequency shift —
/// each separated by normal operation to show clean detection and recovery.
///
/// To build and run:
///   cmake -B build -DCMAKE_BUILD_TYPE=Release
///   cmake --build build
///   ./build/StreamingAnomaly              (default: raw features)
///   ./build/StreamingAnomaly raw          (explicit raw)
///   ./build/StreamingAnomaly translation  (translation 2.5N features)

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include "ESN.h"
#include "Reservoir.h"
#include "TranslationLayer.h"

// --- Signal generation ---
// Multi-harmonic process: fundamental + 3rd harmonic.
// freq_mult scales the fundamental frequency (1.0 = normal).
static void GenerateProcess(float* out, size_t n, size_t t_start,
                             float noise_level, float dc_drift, float freq_mult,
                             std::mt19937_64& rng)
{
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    for (size_t t = 0; t < n; ++t)
    {
        float phase = 0.1f * freq_mult * static_cast<float>(t_start + t);
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
static double PredictBatch(const LinearReadout& readout, const float* features,
                            const float* targets, size_t n, size_t num_features)
{
    std::vector<float> pred(n);
    for (size_t i = 0; i < n; ++i)
        pred[i] = readout.PredictRaw(features + i * num_features);
    return ComputeRMSE(pred.data(), targets, n);
}

// Anomaly event descriptor
struct Event
{
    const char* label;
    float noise;
    float drift;
    float freq;
};

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

    constexpr size_t DIM = 8;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t FEATURES_TRANS = TranslationFeatureCount<DIM>();

    constexpr size_t warmup = 500;
    constexpr size_t prime_steps = 4000;
    constexpr size_t window = 200;
    constexpr float normal_noise = 0.01f;
    constexpr float anomaly_threshold = 5.0f;  // Alert if RMSE > 5x baseline

    const uint64_t seed = 42;
    std::mt19937_64 signal_rng(seed + 777);
    const size_t num_features = use_translation ? FEATURES_TRANS : N;

    // Define the monitoring scenario: 30 windows with 3 anomaly events
    // Each event is bracketed by normal operation to show detection + recovery.
    //                         label              noise  drift  freq
    const Event normal    = { "Normal     ",      0.01f, 0.0f,  1.0f };
    const Event spike     = { "Noise spike",      0.12f, 0.0f,  1.0f };
    const Event drift_evt = { "DC drift   ",      0.01f, 0.30f, 1.0f };
    const Event freq_evt  = { "Freq shift ",      0.01f, 0.0f,  1.3f };

    // Window schedule: normal, anomaly, normal, anomaly, normal, anomaly, normal
    std::vector<Event> schedule;
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);     // 1-5:   normal
    for (size_t i = 0; i < 3; ++i) schedule.push_back(spike);      // 6-8:   noise spike
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);     // 9-13:  recovery
    for (size_t i = 0; i < 3; ++i) schedule.push_back(drift_evt);  // 14-16: DC drift
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);     // 17-21: recovery
    for (size_t i = 0; i < 3; ++i) schedule.push_back(freq_evt);   // 22-24: freq shift
    for (size_t i = 0; i < 6; ++i) schedule.push_back(normal);     // 25-30: recovery (extra window for state washout)

    std::cout << "=== HypercubeRC: Streaming Anomaly Detection ===\n\n";
    std::cout << "Mode: " << (use_translation ? "translation (2.5N)" : "raw (N)") << "\n";
    std::cout << "Signal: 0.6*sin(0.1t) + 0.2*sin(0.3t) + noise + drift\n";
    std::cout << "Anomaly threshold: " << anomaly_threshold << "x baseline RMSE\n";
    std::cout << "DIM=" << DIM << "  N=" << N << "  Features=" << num_features << "\n";
    std::cout << "Usage: " << argv[0] << " [raw|translation]\n\n";

    std::cout << "Anomaly events:\n";
    std::cout << "  1. Noise spike:  noise 0.01 -> 0.12 (12x)\n";
    std::cout << "  2. DC drift:     +0.30 offset\n";
    std::cout << "  3. Freq shift:   frequency x1.3\n\n";

    // =================================================================
    // PHASE 1: PRIME on normal operation
    // =================================================================
    std::cout << "--- Phase 1: Prime on normal operation ---\n";

    size_t t_global = 0;
    std::vector<float> prime_signal(warmup + prime_steps + 1);
    GenerateProcess(prime_signal.data(), prime_signal.size(), t_global,
                     normal_noise, 0.0f, 1.0f, signal_rng);

    auto mode = use_translation ? FeatureMode::Translation : FeatureMode::Raw;
    ESN<DIM> esn(seed, ReadoutType::Linear, mode);

    esn.Warmup(prime_signal.data(), warmup);
    esn.Run(prime_signal.data() + warmup, prime_steps);
    t_global += warmup + prime_steps;

    // Get features
    const float* prime_feat_ptr;
    std::vector<float> prime_translated;
    if (use_translation)
    {
        prime_translated = TranslationTransform<DIM>(esn.States(), prime_steps);
        prime_feat_ptr = prime_translated.data();
    }
    else
    {
        prime_feat_ptr = esn.States();
    }

    std::vector<float> prime_targets(prime_steps);
    for (size_t t = 0; t < prime_steps; ++t)
        prime_targets[t] = prime_signal[warmup + t + 1];

    size_t train_n = static_cast<size_t>(prime_steps * 0.7);
    size_t test_n = prime_steps - train_n;

    LinearReadout readout;
    readout.Train(prime_feat_ptr, prime_targets.data(), train_n, num_features);

    double baseline = PredictBatch(readout, prime_feat_ptr + train_n * num_features,
                                    prime_targets.data() + train_n, test_n, num_features);
    double threshold = baseline * anomaly_threshold;

    std::cout << "  Trained on " << prime_steps << " samples\n";
    std::cout << "  Baseline RMSE: " << std::fixed << std::setprecision(6) << baseline << "\n";
    std::cout << "  Anomaly threshold: " << std::setprecision(6) << threshold << "\n\n";

    // =================================================================
    // PHASE 2: STREAMING MONITOR
    // =================================================================
    std::cout << "--- Phase 2: Streaming monitor (" << schedule.size()
              << " windows of " << window << " steps) ---\n\n";

    std::cout << "  Window | Condition   |     RMSE     | Ratio | Status\n";
    std::cout << "  -------+-------------+--------------+-------+------------------\n";

    for (size_t w = 0; w < schedule.size(); ++w)
    {
        const Event& evt = schedule[w];

        esn.ClearStates();
        std::vector<float> sig(window + 1);
        GenerateProcess(sig.data(), sig.size(), t_global,
                         evt.noise, evt.drift, evt.freq, signal_rng);

        esn.Run(sig.data(), window);
        t_global += window;

        // Get features
        const float* feat_ptr;
        std::vector<float> win_translated;
        if (use_translation)
        {
            win_translated = TranslationTransform<DIM>(esn.States(), window);
            feat_ptr = win_translated.data();
        }
        else
        {
            feat_ptr = esn.States();
        }

        std::vector<float> tgt(window);
        for (size_t t = 0; t < window; ++t)
            tgt[t] = sig[t + 1];

        double rmse = PredictBatch(readout, feat_ptr, tgt.data(), window, num_features);
        double ratio = rmse / baseline;

        const char* status = "";
        if (rmse > threshold)
            status = "** ANOMALY **";

        std::cout << "  " << std::setw(5) << (w + 1)
                  << "  | " << evt.label
                  << " | " << std::setprecision(6) << std::setw(12) << rmse
                  << " | " << std::setprecision(1) << std::setw(5) << ratio
                  << " | " << status << "\n";
    }

    std::cout << "\n--- Summary ---\n";
    std::cout << "  The model learns normal process dynamics during priming, then monitors\n";
    std::cout << "  prediction error against a " << anomaly_threshold << "x baseline threshold.\n\n";
    std::cout << "  Three anomaly types are injected, each for 3 windows:\n";
    std::cout << "    - Noise spike (12x):  prediction error rises from random disturbance\n";
    std::cout << "    - DC drift (+0.30):   systematic offset the model didn't learn\n";
    std::cout << "    - Freq shift (x1.3):  changed dynamics break the learned pattern\n\n";
    std::cout << "  After each event, normal operation resumes and RMSE returns to baseline,\n";
    std::cout << "  confirming the reservoir's state recovers and the readout remains valid.\n";
    std::cout << "  Note: the first recovery window after freq shift may show elevated error\n";
    std::cout << "  as the reservoir washes out residual dynamics from the changed frequency.\n";

    return 0;
}
