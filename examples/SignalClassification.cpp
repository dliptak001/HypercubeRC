/// @file SignalClassification.cpp
/// @brief Classify waveform types from reservoir states.
///
/// The reservoir acts as a feature extractor for pattern recognition. Four
/// waveform types — sine, square, triangle, chirp — cycle in alternating blocks.
/// One-vs-rest readouts classify each timestep by waveform type using only the
/// reservoir state, with per-class accuracy, confusion matrix, and transition
/// dynamics analysis.
///
/// NOTE: Multi-class one-vs-rest classification trains 4 independent readouts,
/// which doesn't fit the single-readout ESN pipeline. ESN handles reservoir
/// driving and feature extraction; readouts are created externally.
///
/// See SignalClassification.md for a detailed walkthrough, expected output, and
/// suggested experiments.

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

int main(int argc, char* argv[])
{
    // --- Parse feature mode ---
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

    constexpr uint64_t seed = 6437149480297576047ULL;
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

    // --- Step 2: Drive the reservoir ---
    ReservoirConfig cfg;
    cfg.seed = seed;
    cfg.leak_rate        = 0.35f;
    cfg.output_fraction  = 0.7f;
    ESN<DIM> esn(cfg, ReadoutType::Ridge, feature_mode);
    const size_t num_features = esn.NumFeatures();

    bool use_translation = (feature_mode == FeatureMode::Translated);
    const char* readout_label = (esn.GetReadoutType() == ReadoutType::Ridge) ? "Ridge" : "Linear";
    std::cout << "Config: DIM=" << DIM << "  N=" << N << "  Outputs=" << esn.NumOutputVerts()
              << " (" << static_cast<int>(esn.OutputFraction() * 100) << "%)"
              << "  Features=" << num_features
              << " (" << (use_translation ? "translation" : "raw") << ")"
              << "  Readout=" << readout_label
              << "  Cycles=" << num_cycles << "  Total=" << collect << " steps\n\n";

    esn.Warmup(signal.data(), warmup);
    esn.Run(signal.data() + warmup, collect);
    esn.EnsureFeatures();
    const float* features = esn.Features();

    // --- Step 3: Train one-vs-rest readouts ---
    size_t train_size = static_cast<size_t>(collect * train_fraction);
    size_t test_size = collect - train_size;

    std::vector<std::vector<float>> class_targets(NUM_CLASSES, std::vector<float>(collect));
    for (size_t t = 0; t < collect; ++t)
        for (size_t c = 0; c < NUM_CLASSES; ++c)
            class_targets[c][t] = (labels[t] == c) ? 1.0f : -1.0f;

    const float* test_features = features + train_size * num_features;
    const size_t* test_labels = labels.data() + train_size;
    std::vector<size_t> predictions(test_size);

    // Multi-class: 4 independent readouts (doesn't fit single-readout ESN pipeline)
    auto train_and_predict = [&](auto& readouts)
    {
        for (size_t c = 0; c < NUM_CLASSES; ++c)
            readouts[c].Train(features, class_targets[c].data(), train_size, num_features);

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
            predictions[t] = predicted;
        }
    };

    if (esn.GetReadoutType() == ReadoutType::Ridge)
    {
        std::vector<RidgeRegression> readouts(NUM_CLASSES);
        train_and_predict(readouts);
    }
    else
    {
        std::vector<LinearReadout> readouts(NUM_CLASSES);
        train_and_predict(readouts);
    }

    std::cout << "--- Training ---\n";
    std::cout << "Reservoir driven through " << collect << " timesteps.\n";
    std::cout << "Trained " << NUM_CLASSES << " one-vs-rest " << readout_label << " readouts on "
              << train_size << " samples (" << num_features << " features each).\n\n";

    // --- Step 4: Evaluate on test set ---
    size_t confusion[NUM_CLASSES][NUM_CLASSES] = {};
    size_t correct = 0;
    for (size_t t = 0; t < test_size; ++t)
    {
        confusion[test_labels[t]][predictions[t]]++;
        if (test_labels[t] == predictions[t])
            correct++;
    }

    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(test_size);

    std::cout << "--- Results (test set: " << test_size << " samples) ---\n\n";
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

    std::cout << "\n--- How fast does the reservoir lock on? ---\n\n";
    std::cout << "When the waveform switches, the reservoir state still reflects the\n";
    std::cout << "previous signal. How many steps until it locks on to the new one?\n\n";

    std::cout << "  Steps after switch  | Accuracy\n";
    std::cout << "  --------------------+---------\n";

    constexpr size_t margins[] = {3, 5, 10, 20, block_size};
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

    std::cout << "\nThe reservoir needs ~20 steps to wash out the old dynamics.\n";
    std::cout << "Steady-state accuracy (step 20+) approaches 100%.\n";

    return 0;
}
