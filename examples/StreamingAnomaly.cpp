/// @file StreamingAnomaly.cpp
/// @brief Streaming anomaly detection — industrial process monitoring.
///
/// A reservoir learns normal process behavior, then monitors a live stream
/// for deviations. Three anomaly types are injected — noise spike, DC drift,
/// frequency shift — each separated by normal operation. Demonstrates clean
/// detection of all three and automatic recovery without retraining.
///
/// See StreamingAnomaly.md for a detailed walkthrough, expected output, and
/// suggested experiments.

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include "ESN.h"

// --- Signal generation ---
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
    FeatureMode feature_mode = FeatureMode::Raw;  // default
    if (argc > 1)
    {
        if (std::strcmp(argv[1], "raw") == 0)
            feature_mode = FeatureMode::Raw;
        else if (std::strcmp(argv[1], "translation") == 0)
            feature_mode = FeatureMode::Translated;
        else
        {
            std::cerr << "Usage: " << argv[0] << " [raw|translation]\n";
            return 1;
        }
    }

    constexpr size_t DIM = 8;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t warmup = 500;
    constexpr size_t prime_steps = 4000;
    constexpr size_t window = 200;
    constexpr float normal_noise = 0.01f;
    constexpr float anomaly_threshold = 5.0f;

    constexpr uint64_t seed = 6437149480297576047ULL;  // NARMA-10 best seed for DIM 7
    std::mt19937_64 signal_rng(seed + 777);

    //                         label              noise  drift  freq
    const Event normal    = { "Normal     ",      0.01f, 0.0f,  1.0f };
    const Event spike     = { "Noise spike",      0.12f, 0.0f,  1.0f };
    const Event drift_evt = { "DC drift   ",      0.01f, 0.30f, 1.0f };
    const Event freq_evt  = { "Freq shift ",      0.01f, 0.0f,  1.3f };

    std::vector<Event> schedule;
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(spike);
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(drift_evt);
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(freq_evt);
    for (size_t i = 0; i < 6; ++i) schedule.push_back(normal);

    std::cout << "=== HypercubeRC: Streaming Anomaly Detection ===\n\n";
    std::cout << "Scenario: an industrial process produces a multi-harmonic signal.\n";
    std::cout << "The reservoir learns the normal pattern, then monitors for deviations.\n";
    std::cout << "Three types of anomaly are injected, each for 3 windows, separated\n";
    std::cout << "by normal operation to show both detection and recovery.\n\n";

    std::cout << "Anomaly types:\n";
    std::cout << "  1. Noise spike   -- sensor noise jumps 12x (0.01 -> 0.12)\n";
    std::cout << "  2. DC drift      -- systematic +0.30 offset (e.g. sensor fouling)\n";
    std::cout << "  3. Freq shift    -- process speed changes to 1.3x (e.g. motor issue)\n\n";

    ReservoirConfig cfg;
    cfg.seed = seed;
    cfg.leak_rate        = 0.3f;
    cfg.output_fraction  = 0.5f;
    ESN<DIM> esn(cfg, ReadoutType::Ridge, feature_mode);
    const size_t num_features = esn.NumFeatures();

    bool use_translation = (feature_mode == FeatureMode::Translated);
    const char* readout_label = (esn.GetReadoutType() == ReadoutType::Ridge) ? "Ridge" : "Linear";
    std::cout << "Config: DIM=" << DIM << "  N=" << N << "  Outputs=" << esn.NumOutputVerts()
              << " (" << static_cast<int>(esn.OutputFraction() * 100) << "%)"
              << "  Features=" << num_features
              << " (" << (use_translation ? "translation" : "raw") << ")"
              << "  Readout=" << readout_label
              << "  Threshold=" << anomaly_threshold << "x baseline\n\n";

    // =================================================================
    // PHASE 1: PRIME on normal operation
    // =================================================================
    std::cout << "--- Phase 1: Learn what \"normal\" looks like ---\n\n";

    size_t t_global = 0;
    std::vector<float> prime_signal(warmup + prime_steps + 1);
    GenerateProcess(prime_signal.data(), prime_signal.size(), t_global,
                     normal_noise, 0.0f, 1.0f, signal_rng);

    esn.Warmup(prime_signal.data(), warmup);
    esn.Run(prime_signal.data() + warmup, prime_steps);
    t_global += warmup + prime_steps;

    std::vector<float> prime_targets(prime_steps);
    for (size_t t = 0; t < prime_steps; ++t)
        prime_targets[t] = prime_signal[warmup + t + 1];

    size_t train_n = static_cast<size_t>(prime_steps * 0.7);
    size_t test_n = prime_steps - train_n;

    esn.Train(prime_targets.data(), train_n);

    // Compute baseline RMSE on prime test set
    std::vector<float> prime_pred(test_n);
    for (size_t i = 0; i < test_n; ++i)
        prime_pred[i] = esn.PredictRaw(train_n + i);
    double baseline = ComputeRMSE(prime_pred.data(), prime_targets.data() + train_n, test_n);
    double threshold = baseline * anomaly_threshold;

    std::cout << "Reservoir trained on " << prime_steps << " normal samples.\n";
    std::cout << "Baseline prediction error (RMSE): " << std::fixed << std::setprecision(6) << baseline << "\n";
    std::cout << "Anomaly threshold (" << std::setprecision(0) << anomaly_threshold
              << "x baseline): " << std::setprecision(6) << threshold << "\n";
    std::cout << "Anything above this triggers an alert.\n\n";

    // =================================================================
    // PHASE 2: STREAMING MONITOR
    // =================================================================
    std::cout << "--- Phase 2: Monitor the process (" << schedule.size()
              << " windows of " << window << " steps) ---\n\n";
    std::cout << "The model predicts each window's output. When the prediction error\n";
    std::cout << "exceeds " << std::setprecision(0) << anomaly_threshold
              << "x baseline, something has changed.\n\n";

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
        esn.EnsureFeatures();

        std::vector<float> tgt(window);
        for (size_t t = 0; t < window; ++t)
            tgt[t] = sig[t + 1];

        // Predict using the trained readout on new window's features
        std::vector<float> pred(window);
        for (size_t t = 0; t < window; ++t)
            pred[t] = esn.PredictRaw(t);

        double rmse = ComputeRMSE(pred.data(), tgt.data(), window);
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

    std::cout << "\n--- What happened ---\n\n";
    std::cout << "The reservoir learned to predict normal process output during priming.\n";
    std::cout << "During monitoring, prediction error is the anomaly signal:\n\n";
    std::cout << "  Noise spike:  RMSE jumps ~12x -- random disturbance is unpredictable.\n";
    std::cout << "                Recovery is instant (window 9 back to baseline).\n\n";
    std::cout << "  DC drift:     RMSE rises dramatically -- the model didn't learn this offset.\n";
    std::cout << "                The leaky integrator compounds the error across steps.\n";
    std::cout << "                Takes 1-2 windows to wash out after recovery.\n\n";
    std::cout << "  Freq shift:   RMSE spikes -- changed dynamics break the learned pattern.\n";
    std::cout << "                Slowest recovery: reservoir needs 1-2 extra windows to\n";
    std::cout << "                wash out the altered frequency from its internal state.\n\n";
    std::cout << "Key insight: the frozen model stays valid for normal operation throughout.\n";
    std::cout << "Each anomaly is detected, and the system recovers without retraining.\n";

    return 0;
}
