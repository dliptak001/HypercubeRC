/// @file StreamingAnomaly.cpp
/// @brief Streaming anomaly detection — industrial process monitoring.
///
/// A reservoir learns normal process behavior, then monitors a live stream
/// for deviations. Three anomaly types are injected — noise spike, DC drift,
/// frequency shift — each separated by normal operation. Two readouts run
/// side-by-side on the same reservoir for comparison:
///   - Ridge: closed-form on stride-selected features (cheap, well suited
///            to streaming since it can be re-primed quickly)
///   - HCNN:  learned CNN on raw reservoir state (frozen after priming —
///            HCNN is batch-only, not incremental)
///
/// Both are trained once in Phase 1 and used frozen during Phase 2
/// monitoring, so the apples-to-apples comparison is "which readout
/// produces a cleaner anomaly signal from the same reservoir dynamics."
///
/// See StreamingAnomaly.md for a detailed walkthrough, expected output, and
/// suggested experiments.

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include "ESN.h"
#include "readout/HCNNPresets.h"

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

    // Common reservoir parameters — applied to both Ridge and HCNN ESNs.
    ReservoirConfig base_cfg;
    base_cfg.seed = seed;
    base_cfg.leak_rate = 0.3f;

    // Ridge ESN: 50% output, Ridge closed-form regression.
    ReservoirConfig ridge_cfg = base_cfg;
    ridge_cfg.output_fraction = 0.5f;
    ESN<DIM> esn_ridge(ridge_cfg, ReadoutType::Ridge, feature_mode);

    // HCNN ESN: 100% output (HCNN always operates on raw state).
    ReservoirConfig hcnn_cfg_r = base_cfg;
    hcnn_cfg_r.output_fraction = 1.0f;
    ESN<DIM> esn_hcnn(hcnn_cfg_r, ReadoutType::HCNN);

    bool use_translation = (feature_mode == FeatureMode::Translated);
    std::cout << "Config: DIM=" << DIM << "  N=" << N
              << "  Leak=" << base_cfg.leak_rate
              << "  Threshold=" << anomaly_threshold << "x baseline\n";
    std::cout << "  Ridge: Outputs=" << esn_ridge.NumOutputVerts()
              << " (" << static_cast<int>(esn_ridge.OutputFraction() * 100) << "%)"
              << "  Features=" << esn_ridge.NumFeatures()
              << " (" << (use_translation ? "translation" : "raw") << ")\n";
    std::cout << "  HCNN : Outputs=" << esn_hcnn.NumOutputVerts()
              << " (100%)  raw state\n\n";

    // =================================================================
    // PHASE 1: PRIME on normal operation
    // =================================================================
    std::cout << "--- Phase 1: Learn what \"normal\" looks like ---\n\n";

    size_t t_global = 0;
    std::vector<float> prime_signal(warmup + prime_steps + 1);
    GenerateProcess(prime_signal.data(), prime_signal.size(), t_global,
                     normal_noise, 0.0f, 1.0f, signal_rng);

    // Drive both reservoirs with the same signal.
    esn_ridge.Warmup(prime_signal.data(), warmup);
    esn_ridge.Run(prime_signal.data() + warmup, prime_steps);

    esn_hcnn.Warmup(prime_signal.data(), warmup);
    esn_hcnn.Run(prime_signal.data() + warmup, prime_steps);

    t_global += warmup + prime_steps;

    std::vector<float> prime_targets(prime_steps);
    for (size_t t = 0; t < prime_steps; ++t)
        prime_targets[t] = prime_signal[warmup + t + 1];

    size_t train_n = static_cast<size_t>(prime_steps * 0.7);
    size_t test_n = prime_steps - train_n;

    // --- Train Ridge ---
    std::cout << "Ridge: training on " << train_n << " samples..." << std::flush;
    auto rt0 = std::chrono::steady_clock::now();
    esn_ridge.Train(prime_targets.data(), train_n);
    auto rt1 = std::chrono::steady_clock::now();
    double ridge_train_s = std::chrono::duration<double>(rt1 - rt0).count();
    std::cout << " done (" << std::fixed << std::setprecision(2) << ridge_train_s << "s)\n";

    // --- Train HCNN ---
    // HRCCNN baseline architecture (nl=1, ch=8, FLAT, lr=0.0015,
    // bs=1<<(DIM-1)) with smooth-signal epochs: ep=25 is the saturation
    // point for the smooth anomaly process here.  The baseline's default
    // ep=2000 is calibrated for chaotic MG/NARMA.
    CNNReadoutConfig cnn_cfg = hcnn_presets::HRCCNNBaseline<DIM>();
    cnn_cfg.num_outputs = 1;
    cnn_cfg.task        = HCNNTask::Regression;
    cnn_cfg.epochs      = 1000;
    cnn_cfg.seed        = 420607;

    std::cout << "HCNN : training on " << train_n << " samples ("
              << cnn_cfg.epochs << " epochs, batch=" << cnn_cfg.batch_size
              << ", lr_max=" << std::setprecision(4) << cnn_cfg.lr_max << ")..." << std::flush;
    auto ht0 = std::chrono::steady_clock::now();
    esn_hcnn.Train(prime_targets.data(), train_n, cnn_cfg);
    auto ht1 = std::chrono::steady_clock::now();
    double hcnn_train_s = std::chrono::duration<double>(ht1 - ht0).count();
    std::cout << " done (" << std::setprecision(2) << hcnn_train_s << "s)\n\n";

    // --- Baseline RMSE for each readout on the held-out prime test set ---
    std::vector<float> ridge_prime_pred(test_n);
    std::vector<float> hcnn_prime_pred(test_n);
    for (size_t i = 0; i < test_n; ++i) {
        ridge_prime_pred[i] = esn_ridge.PredictRaw(train_n + i);
        hcnn_prime_pred[i]  = esn_hcnn.PredictRaw(train_n + i);
    }
    double baseline_ridge = ComputeRMSE(ridge_prime_pred.data(),
                                        prime_targets.data() + train_n, test_n);
    double baseline_hcnn  = ComputeRMSE(hcnn_prime_pred.data(),
                                        prime_targets.data() + train_n, test_n);
    double threshold_ridge = baseline_ridge * anomaly_threshold;
    double threshold_hcnn  = baseline_hcnn  * anomaly_threshold;

    std::cout << "Baseline (prime test, RMSE):\n";
    std::cout << "  Ridge: " << std::setprecision(6) << baseline_ridge
              << "   threshold " << threshold_ridge << "\n";
    std::cout << "  HCNN : " << baseline_hcnn
              << "   threshold " << threshold_hcnn << "\n\n";

    // =================================================================
    // PHASE 2: STREAMING MONITOR (both readouts in parallel)
    // =================================================================
    std::cout << "--- Phase 2: Monitor the process (" << schedule.size()
              << " windows of " << window << " steps) ---\n\n";
    std::cout << "Each window is fed to the reservoir, and both readouts predict\n";
    std::cout << "the next value.  An RMSE above " << std::setprecision(0) << anomaly_threshold
              << "x that readout's baseline is flagged.\n";
    std::cout << "Status column: 'R' = Ridge anomaly, 'H' = HCNN anomaly.\n\n";

    std::cout << "                              |       Ridge       |        HCNN       |\n";
    std::cout << "  Window | Condition          |    RMSE     Ratio |    RMSE     Ratio | Status\n";
    std::cout << "  -------+--------------------+-------------------+-------------------+---------\n";

    size_t ridge_flags = 0, hcnn_flags = 0;

    for (size_t w = 0; w < schedule.size(); ++w)
    {
        const Event& evt = schedule[w];

        std::vector<float> sig(window + 1);
        GenerateProcess(sig.data(), sig.size(), t_global,
                         evt.noise, evt.drift, evt.freq, signal_rng);
        t_global += window;

        std::vector<float> tgt(window);
        for (size_t t = 0; t < window; ++t)
            tgt[t] = sig[t + 1];

        // --- Ridge path ---
        esn_ridge.ClearStates();
        esn_ridge.Run(sig.data(), window);
        esn_ridge.EnsureFeatures();

        std::vector<float> ridge_pred(window);
        for (size_t t = 0; t < window; ++t)
            ridge_pred[t] = esn_ridge.PredictRaw(t);
        double ridge_rmse = ComputeRMSE(ridge_pred.data(), tgt.data(), window);
        double ridge_ratio = ridge_rmse / baseline_ridge;
        bool ridge_anom = (ridge_rmse > threshold_ridge);
        if (ridge_anom) ++ridge_flags;

        // --- HCNN path ---
        esn_hcnn.ClearStates();
        esn_hcnn.Run(sig.data(), window);

        std::vector<float> hcnn_pred(window);
        for (size_t t = 0; t < window; ++t)
            hcnn_pred[t] = esn_hcnn.PredictRaw(t);
        double hcnn_rmse = ComputeRMSE(hcnn_pred.data(), tgt.data(), window);
        double hcnn_ratio = hcnn_rmse / baseline_hcnn;
        bool hcnn_anom = (hcnn_rmse > threshold_hcnn);
        if (hcnn_anom) ++hcnn_flags;

        char status[16] = "";
        if (ridge_anom && hcnn_anom)      std::snprintf(status, sizeof(status), "** R+H **");
        else if (ridge_anom)              std::snprintf(status, sizeof(status), "** R **");
        else if (hcnn_anom)               std::snprintf(status, sizeof(status), "** H **");

        std::cout << "  " << std::setw(5) << (w + 1)
                  << "  | " << evt.label << "        "
                  << " | " << std::fixed << std::setprecision(6) << std::setw(10) << ridge_rmse
                  << "  " << std::setprecision(1) << std::setw(5) << ridge_ratio
                  << " | " << std::setprecision(6) << std::setw(10) << hcnn_rmse
                  << "  " << std::setprecision(1) << std::setw(5) << hcnn_ratio
                  << " | " << status << "\n";
    }

    std::cout << "\nFlagged windows: Ridge=" << ridge_flags
              << "  HCNN=" << hcnn_flags
              << "  (expected 11 = 9 anomaly windows + 2 washout windows\n"
              << "                         where the leaky integrator is still ringing)\n\n";

    std::cout << "--- What happened ---\n\n";
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
    std::cout << "Note: HCNN is frozen after priming -- it does not support incremental\n";
    std::cout << "updates, so it monitors in the same way as Ridge here.  For applications\n";
    std::cout << "where slow drift must be tracked online, use LinearReadout with\n";
    std::cout << "TrainIncremental instead (see docs/Readout.md).\n";

    return 0;
}
