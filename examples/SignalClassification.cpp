/// @file SignalClassification.cpp
/// @brief Classify waveform types from reservoir states (Ridge vs HCNN).
///
/// The reservoir acts as a feature extractor for pattern recognition. Four
/// waveform types -- sine, square, triangle, chirp -- cycle in alternating blocks.
/// Two classifiers run side-by-side on the same reservoir:
///   - Ridge: 4 one-vs-rest readouts (one per class), argmax prediction
///   - HCNN:  single multi-class readout (num_outputs=4, softmax over classes)
///
/// See SignalClassification.md for a detailed walkthrough, expected output, and
/// suggested experiments.

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include "ESN.h"

static constexpr float PI = 3.14159265358979323846f;
static constexpr size_t NUM_CLASSES = 4;
static const char* CLASS_NAMES[NUM_CLASSES] = {"Sine    ", "Square  ", "Triangle", "Chirp   "};

// Each waveform has a distinct frequency and dynamic character.
static float GenerateWaveform(size_t waveform, float phase)
{
    switch (waveform)
    {
    case 0: // Sine — slow, smooth (freq 0.08)
        return std::sin(phase);

    case 1: // Square — fast, discontinuous (freq 0.25)
        return std::sin(phase) >= 0.0f ? 0.9f : -0.9f;

    case 2: // Triangle — medium, piecewise linear (freq 0.15)
    {
        float p = std::fmod(phase, 2.0f * PI);
        if (p < 0) p += 2.0f * PI;
        return (p < PI) ? (-1.0f + 2.0f * p / PI) : (3.0f - 2.0f * p / PI);
    }

    case 3: // Chirp — accelerating frequency
        return std::sin(phase + 0.3f * phase * phase);

    default:
        return 0.0f;
    }
}

static constexpr float CLASS_FREQ[NUM_CLASSES] = { 0.08f, 0.25f, 0.15f, 0.10f };

/// @brief Compute and print accuracy, per-class breakdown, confusion matrix,
///        and lock-on dynamics for a given set of predictions.
/// @return Overall accuracy (0-100%)
static double analyze_and_print(const char* label,
                                const std::vector<size_t>& predictions,
                                const size_t* test_labels,
                                size_t test_size,
                                size_t train_size,
                                size_t block_size)
{
    std::cout << "--- " << label << " results (test set: " << test_size << " samples) ---\n\n";

    size_t confusion[NUM_CLASSES][NUM_CLASSES] = {};
    size_t correct = 0;
    for (size_t t = 0; t < test_size; ++t)
    {
        confusion[test_labels[t]][predictions[t]]++;
        if (test_labels[t] == predictions[t])
            correct++;
    }

    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(test_size);

    std::cout << "Overall accuracy: " << std::fixed << std::setprecision(1)
              << accuracy << "%\n\n";

    std::cout << "Per-class breakdown:\n";
    for (size_t c = 0; c < NUM_CLASSES; ++c)
    {
        size_t total_c = 0;
        for (size_t p = 0; p < NUM_CLASSES; ++p)
            total_c += confusion[c][p];
        double acc = (total_c > 0) ? 100.0 * confusion[c][c] / total_c : 0.0;
        std::cout << "  " << CLASS_NAMES[c] << "  " << std::setprecision(1)
                  << std::setw(5) << acc << "%  (" << confusion[c][c]
                  << "/" << total_c << ")";

        if (acc >= 99.9)
            std::cout << "  -- perfectly separable";
        else if (acc >= 98.0)
            std::cout << "  -- near-perfect";
        else if (acc < 90.0)
        {
            size_t max_conf = 0;
            size_t max_conf_class = c;
            for (size_t p = 0; p < NUM_CLASSES; ++p)
            {
                if (p != c && confusion[c][p] > max_conf)
                {
                    max_conf = confusion[c][p];
                    max_conf_class = p;
                }
            }
            if (max_conf > 0)
            {
                double conf_pct = 100.0 * max_conf / total_c;
                std::cout << "  -- confused with " << CLASS_NAMES[max_conf_class]
                          << " " << std::setprecision(0) << conf_pct << "% of the time";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nConfusion matrix (rows=actual, cols=predicted):\n";
    std::cout << "               ";
    for (size_t c = 0; c < NUM_CLASSES; ++c)
        std::cout << CLASS_NAMES[c] << "  ";
    std::cout << "\n";

    for (size_t a = 0; a < NUM_CLASSES; ++a)
    {
        size_t total_a = 0;
        for (size_t p = 0; p < NUM_CLASSES; ++p)
            total_a += confusion[a][p];

        std::cout << "  " << CLASS_NAMES[a] << " |";
        for (size_t p = 0; p < NUM_CLASSES; ++p)
        {
            double pct = (total_a > 0) ? 100.0 * confusion[a][p] / total_a : 0.0;
            std::cout << std::setw(7) << std::setprecision(1) << pct << "%  ";
        }
        std::cout << "\n";
    }

    std::cout << "\nLock-on dynamics (steps after block transition):\n";
    std::cout << "  Steps after switch  | Accuracy\n";
    std::cout << "  --------------------+---------\n";

    const size_t margins[] = {3, 5, 10, 20, block_size};
    for (size_t margin : margins)
    {
        size_t ok = 0, total = 0;

        for (size_t t = 0; t < test_size; ++t)
        {
            size_t global_t = train_size + t;
            size_t pos_in_block = global_t % block_size;

            if (pos_in_block < margin)
            {
                total++;
                if (predictions[t] == test_labels[t]) ok++;
            }
        }

        double acc = (total > 0) ? 100.0 * ok / total : 0.0;
        if (margin == block_size)
            std::cout << "  Entire block        |  " << std::setprecision(1)
                      << std::setw(5) << acc << "%  (overall)\n";
        else
            std::cout << "  0 - " << std::setw(2) << margin
                      << "              |  " << std::setprecision(1)
                      << std::setw(5) << acc << "%\n";
    }
    std::cout << "\n";
    return accuracy;
}

int main(int argc, char* argv[])
{
    // --- Parse feature mode (applies to Ridge path only) ---
    FeatureMode feature_mode = FeatureMode::Translated;  // default — classification needs translation
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

    // --- Configuration ---
    constexpr size_t DIM = 7;
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t warmup = 300;
    constexpr size_t block_size = 150;
    constexpr size_t num_cycles = 20;
    constexpr size_t collect = block_size * NUM_CLASSES * num_cycles;
    constexpr double train_fraction = 0.7;

    constexpr uint64_t seed = 6437149480297576047ULL;  // NARMA-10 best seed for DIM 7
    std::cout << "=== HypercubeRC: Signal Classification ===\n\n";
    std::cout << "Task: identify which waveform is currently being fed to the reservoir,\n";
    std::cout << "using only the reservoir's internal state -- not the input directly.\n\n";
    std::cout << "Four waveforms cycle in blocks of " << block_size << " steps:\n";
    std::cout << "  Sine (f=0.08)  |  Square (f=0.25)  |  Triangle (f=0.15)  |  Chirp (sweep)\n";
    std::cout << "Each has a distinct frequency and dynamic signature.\n\n";

    // --- Step 1: Generate the input signal and labels ---
    std::vector<float> signal(warmup + collect);
    std::vector<size_t> labels(collect);

    for (size_t t = 0; t < warmup; ++t)
        signal[t] = GenerateWaveform(0, CLASS_FREQ[0] * static_cast<float>(t));

    for (size_t t = 0; t < collect; ++t)
    {
        size_t block_idx = t / block_size;
        size_t waveform = block_idx % NUM_CLASSES;
        size_t t_in_block = t % block_size;

        labels[t] = waveform;
        float phase = CLASS_FREQ[waveform] * static_cast<float>(t_in_block);
        signal[warmup + t] = GenerateWaveform(waveform, phase);
    }

    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;
    const size_t* test_labels = labels.data() + train_size;

    // ===================================================================
    //  Ridge: 4 one-vs-rest readouts
    // ===================================================================
    ReservoirConfig ridge_cfg;
    ridge_cfg.seed = seed;
    ridge_cfg.leak_rate = 0.35f;
    ridge_cfg.output_fraction = 0.7f;
    ESN<DIM> esn_ridge(ridge_cfg, ReadoutType::Ridge, feature_mode);
    const size_t num_features = esn_ridge.NumFeatures();

    bool use_translation = (feature_mode == FeatureMode::Translated);
    std::cout << "Ridge config: DIM=" << DIM << "  N=" << N
              << "  Outputs=" << esn_ridge.NumOutputVerts()
              << " (" << static_cast<int>(esn_ridge.OutputFraction() * 100) << "%)"
              << "  Features=" << num_features
              << " (" << (use_translation ? "translation" : "raw") << ")"
              << "  Cycles=" << num_cycles << "  Total=" << collect << " steps\n\n";

    esn_ridge.Warmup(signal.data(), warmup);
    esn_ridge.Run(signal.data() + warmup, collect);
    esn_ridge.EnsureFeatures();
    const float* features = esn_ridge.Features();

    // Build one-vs-rest targets: +1 for the class, -1 for all others.
    std::vector<std::vector<float>> class_targets(NUM_CLASSES, std::vector<float>(collect));
    for (size_t t = 0; t < collect; ++t)
        for (size_t c = 0; c < NUM_CLASSES; ++c)
            class_targets[c][t] = (labels[t] == c) ? 1.0f : -1.0f;

    // Train 4 Ridge readouts and predict via argmax.
    std::vector<size_t> ridge_predictions(test_size);
    {
        std::vector<RidgeRegression> readouts(NUM_CLASSES);
        auto t0 = std::chrono::steady_clock::now();
        for (size_t c = 0; c < NUM_CLASSES; ++c)
            readouts[c].Train(features, class_targets[c].data(), train_size, num_features);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        const float* test_features = features + train_size * num_features;
        for (size_t t = 0; t < test_size; ++t)
        {
            size_t predicted = 0;
            float best_score = readouts[0].PredictRaw(test_features + t * num_features);
            for (size_t c = 1; c < NUM_CLASSES; ++c)
            {
                float score = readouts[c].PredictRaw(test_features + t * num_features);
                if (score > best_score)
                {
                    best_score = score;
                    predicted = c;
                }
            }
            ridge_predictions[t] = predicted;
        }

        std::cout << "Ridge: trained " << NUM_CLASSES << " one-vs-rest readouts on "
                  << train_size << " samples in " << std::fixed << std::setprecision(3)
                  << secs << "s\n\n";
    }

    double ridge_acc = analyze_and_print("Ridge", ridge_predictions, test_labels,
                                         test_size, train_size, block_size);

    // ===================================================================
    //  HCNN: single multi-class readout
    // ===================================================================
    ReservoirConfig hcnn_cfg_r;
    hcnn_cfg_r.seed = seed;
    hcnn_cfg_r.leak_rate = 0.35f;
    hcnn_cfg_r.output_fraction = 1.0f;  // HCNN uses all vertices
    ESN<DIM> esn_hcnn(hcnn_cfg_r, ReadoutType::HCNN);

    std::cout << "HCNN config: DIM=" << DIM << "  N=" << N
              << "  raw state (all vertices)  Task=Classification  Classes=" << NUM_CLASSES << "\n";

    esn_hcnn.Warmup(signal.data(), warmup);
    esn_hcnn.Run(signal.data() + warmup, collect);

    // HCNN classification takes float class indices, not one-hot.
    std::vector<float> float_labels(collect);
    for (size_t t = 0; t < collect; ++t)
        float_labels[t] = static_cast<float>(labels[t]);

    CNNReadoutConfig cnn_cfg;
    cnn_cfg.num_outputs = NUM_CLASSES;
    cnn_cfg.task = HCNNTask::Classification;
    cnn_cfg.epochs = 25;
    cnn_cfg.batch_size = 128;
    cnn_cfg.lr_max = 0.002f;

    std::cout << "HCNN training: " << cnn_cfg.epochs << " epochs, batch=" << cnn_cfg.batch_size
              << ", lr_max=" << std::setprecision(4) << cnn_cfg.lr_max
              << " (cosine floor " << cnn_cfg.lr_max * cnn_cfg.lr_min_frac << ")\n";
    std::cout << "Training..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn_hcnn.Train(float_labels.data(), train_size, cnn_cfg);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n\n";

    // Per-sample predictions via argmax of the logit vector.
    std::vector<size_t> hcnn_predictions(test_size);
    std::vector<float> logits(NUM_CLASSES);
    for (size_t t = 0; t < test_size; ++t)
    {
        esn_hcnn.PredictRaw(train_size + t, logits.data());
        size_t predicted = 0;
        float best = logits[0];
        for (size_t c = 1; c < NUM_CLASSES; ++c)
        {
            if (logits[c] > best)
            {
                best = logits[c];
                predicted = c;
            }
        }
        hcnn_predictions[t] = predicted;
    }

    double hcnn_acc = analyze_and_print("HCNN", hcnn_predictions, test_labels,
                                        test_size, train_size, block_size);

    // ===================================================================
    //  Comparison
    // ===================================================================
    std::cout << "--- Comparison ---\n\n";
    std::cout << "  Readout |  Accuracy\n";
    std::cout << "  --------+----------\n";
    std::cout << "  Ridge   |  " << std::fixed << std::setprecision(2)
              << std::setw(6) << ridge_acc << "%\n";
    std::cout << "  HCNN    |  " << std::setw(6) << hcnn_acc << "%\n\n";

    std::cout << "Both classifiers use the same reservoir dynamics.  Ridge trains\n";
    std::cout << "4 one-vs-rest regressors on hand-crafted translation features;\n";
    std::cout << "HCNN trains a single multi-class network on raw reservoir state.\n";

    return 0;
}
