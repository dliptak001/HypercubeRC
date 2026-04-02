/// CoreSmokeTest — exercises the full HypercubeRCCore pipeline.
///
/// Links against the static library exactly as an SDK consumer would.
/// Returns 0 on success, 1 on any failure. No external dependencies.

#include "ESN.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

static int failures = 0;

static void check(bool ok, const char* name)
{
    if (ok) {
        printf("  PASS  %s\n", name);
    } else {
        printf("  FAIL  %s\n", name);
        ++failures;
    }
}

// ── Generate sine signal ──
static std::vector<float> make_sine(size_t n, float freq = 1.0f)
{
    std::vector<float> s(n);
    for (size_t i = 0; i < n; ++i)
        s[i] = std::sin(static_cast<float>(i) * freq * 6.2831853f / static_cast<float>(n));
    return s;
}

// ── Test: ESN with Ridge readout (translated features) ──
template <size_t DIM>
void test_ridge_prediction()
{
    constexpr size_t N = 1ULL << DIM;
    std::string label = "DIM " + std::to_string(DIM) + " Ridge prediction (N=" + std::to_string(N) + ")";

    auto signal = make_sine(2000, 10.0f);

    ESN<DIM> esn(ReservoirConfig{.seed = 42}, ReadoutType::Ridge, FeatureMode::Translated);
    esn.Warmup(signal.data(), 200);
    esn.Run(signal.data() + 200, 1799);
    esn.Train(signal.data() + 201, 1400);

    double r2 = esn.R2(signal.data() + 201, 0, esn.NumCollected());
    double nrmse = esn.NRMSE(signal.data() + 201, 0, esn.NumCollected());

    printf("         R2=%.6f  NRMSE=%.6f\n", r2, nrmse);
    check(r2 > 0.99 && nrmse < 0.1, label.c_str());
}

// ── Test: ESN with Linear readout (raw features) ──
template <size_t DIM>
void test_linear_prediction()
{
    constexpr size_t N = 1ULL << DIM;
    std::string label = "DIM " + std::to_string(DIM) + " Linear prediction (N=" + std::to_string(N) + ")";

    auto signal = make_sine(2000, 10.0f);

    ESN<DIM> esn(ReservoirConfig{.seed = 42}, ReadoutType::Linear, FeatureMode::Raw);
    esn.Warmup(signal.data(), 200);
    esn.Run(signal.data() + 200, 1799);
    esn.Train(signal.data() + 201, 1400);

    double r2 = esn.R2(signal.data() + 201, 0, esn.NumCollected());
    printf("         R2=%.6f\n", r2);
    check(r2 > 0.90, label.c_str());
}

// ── Test: Classification with accuracy check ──
template <size_t DIM>
void test_classification()
{
    std::string label = "DIM " + std::to_string(DIM) + " classification";

    // Binary classification: positive vs negative sine half-cycles
    auto signal = make_sine(2000, 10.0f);
    std::vector<float> labels(1799);
    for (size_t i = 0; i < 1799; ++i)
        labels[i] = signal[i + 201] >= 0.0f ? 1.0f : -1.0f;

    ESN<DIM> esn(ReservoirConfig{.seed = 7}, ReadoutType::Ridge, FeatureMode::Translated);
    esn.Warmup(signal.data(), 200);
    esn.Run(signal.data() + 200, 1799);
    esn.Train(labels.data(), 1400);

    double acc = esn.Accuracy(labels.data(), 0, esn.NumCollected());
    printf("         accuracy=%.4f\n", acc);
    check(acc > 0.90, label.c_str());
}

// ── Test: ESN construction across all supported DIMs ──
void test_all_dims_construct()
{
    ESN<5>  e5 (ReservoirConfig{.seed = 1}, ReadoutType::Ridge, FeatureMode::Translated);
    ESN<6>  e6 (ReservoirConfig{.seed = 1}, ReadoutType::Ridge, FeatureMode::Translated);
    ESN<7>  e7 (ReservoirConfig{.seed = 1}, ReadoutType::Ridge, FeatureMode::Translated);
    ESN<8>  e8 (ReservoirConfig{.seed = 1}, ReadoutType::Ridge, FeatureMode::Translated);
    ESN<9>  e9 (ReservoirConfig{.seed = 1}, ReadoutType::Ridge, FeatureMode::Translated);
    ESN<10> e10(ReservoirConfig{.seed = 1}, ReadoutType::Ridge, FeatureMode::Translated);
    ESN<11> e11(ReservoirConfig{.seed = 1}, ReadoutType::Ridge, FeatureMode::Translated);
    ESN<12> e12(ReservoirConfig{.seed = 1}, ReadoutType::Ridge, FeatureMode::Translated);

    bool ok = e5.NumFeatures() > 0 && e6.NumFeatures() > 0 &&
              e7.NumFeatures() > 0 && e8.NumFeatures() > 0 &&
              e9.NumFeatures() > 0 && e10.NumFeatures() > 0 &&
              e11.NumFeatures() > 0 && e12.NumFeatures() > 0;
    check(ok, "all DIMs construct with correct feature counts");
}

// ── Test: Persistence round-trip (GetReadoutState / SetReadoutState) ──
template <size_t DIM>
void test_persistence()
{
    std::string label = "DIM " + std::to_string(DIM) + " persistence round-trip";

    auto signal = make_sine(2000, 10.0f);

    // Train a model
    ESN<DIM> esn1(ReservoirConfig{.seed = 42}, ReadoutType::Ridge, FeatureMode::Translated);
    esn1.Warmup(signal.data(), 200);
    esn1.Run(signal.data() + 200, 1799);
    esn1.Train(signal.data() + 201, 1400);
    float pred_original = esn1.PredictRaw(0);

    // Extract state
    auto state = esn1.GetReadoutState();

    // Create a new ESN, drive with same data, restore state
    ESN<DIM> esn2(ReservoirConfig{.seed = 42}, ReadoutType::Ridge, FeatureMode::Translated);
    esn2.Warmup(signal.data(), 200);
    esn2.Run(signal.data() + 200, 1799);
    esn2.SetReadoutState(state);
    float pred_restored = esn2.PredictRaw(0);

    float diff = std::fabs(pred_original - pred_restored);
    printf("         original=%.6f restored=%.6f diff=%.2e\n",
           pred_original, pred_restored, diff);
    check(diff < 1e-6f, label.c_str());
}

// ── Test: ClearStates preserves trained readout ──
template <size_t DIM>
void test_clear_states()
{
    std::string label = "DIM " + std::to_string(DIM) + " ClearStates preserves readout";

    auto signal = make_sine(2000, 10.0f);

    ESN<DIM> esn(ReservoirConfig{.seed = 42}, ReadoutType::Ridge, FeatureMode::Translated);
    esn.Warmup(signal.data(), 200);
    esn.Run(signal.data() + 200, 1799);
    esn.Train(signal.data() + 201, 1400);

    // Clear and re-run
    esn.ClearStates();
    check(esn.NumCollected() == 0, (label + " — count reset").c_str());

    esn.Run(signal.data() + 200, 1799);
    double r2 = esn.R2(signal.data() + 201, 0, esn.NumCollected());
    printf("         R2 after clear+re-run=%.6f\n", r2);
    check(r2 > 0.99, (label + " — readout intact").c_str());
}

// ── Test: TrainIncremental (streaming, Linear readout only) ──
template <size_t DIM>
void test_train_incremental()
{
    std::string label = "DIM " + std::to_string(DIM) + " TrainIncremental (streaming)";

    auto signal = make_sine(2000, 10.0f);

    ESN<DIM> esn(ReservoirConfig{.seed = 42}, ReadoutType::Linear, FeatureMode::Translated);
    esn.Warmup(signal.data(), 200);
    esn.Run(signal.data() + 200, 1799);

    // Initial batch training
    esn.Train(signal.data() + 201, 1400);
    double r2_before = esn.R2(signal.data() + 201, 0, esn.NumCollected());

    // Incremental update with same data (should not degrade)
    esn.TrainIncremental(signal.data() + 201, 1400, /*blend=*/0.3f);
    double r2_after = esn.R2(signal.data() + 201, 0, esn.NumCollected());

    printf("         R2 before=%.6f after=%.6f\n", r2_before, r2_after);
    check(r2_after > 0.90, label.c_str());
}

// ── Test: Multi-input ESN (num_inputs > 1) ──
template <size_t DIM>
void test_multi_input()
{
    std::string label = "DIM " + std::to_string(DIM) + " multi-input (2 inputs)";

    constexpr size_t K = 2;
    constexpr size_t total_steps = 1800;
    constexpr size_t warmup_steps = 200;
    constexpr size_t run_steps = total_steps - warmup_steps;

    // Two sine inputs at different frequencies
    std::vector<float> inputs(total_steps * K);
    for (size_t i = 0; i < total_steps; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(total_steps);
        inputs[i * K + 0] = std::sin(t * 62.83f);  // ~10 cycles
        inputs[i * K + 1] = std::sin(t * 31.42f);  //  ~5 cycles
    }

    // Target: sum of the two inputs (one step ahead)
    std::vector<float> targets(run_steps);
    for (size_t i = 0; i < run_steps; ++i) {
        size_t idx = warmup_steps + i + 1;
        if (idx < total_steps)
            targets[i] = inputs[idx * K + 0] + inputs[idx * K + 1];
        else
            targets[i] = 0.0f;
    }

    ReservoirConfig cfg{.seed = 42, .num_inputs = K};
    ESN<DIM> esn(cfg, ReadoutType::Ridge, FeatureMode::Translated);
    esn.Warmup(inputs.data(), warmup_steps);
    esn.Run(inputs.data() + warmup_steps * K, run_steps);
    esn.Train(targets.data(), run_steps - 1);  // -1: last target looks ahead past data

    double r2 = esn.R2(targets.data(), 0, run_steps - 1);
    printf("         R2=%.6f\n", r2);
    check(r2 > 0.90, label.c_str());
}

int main()
{
    printf("=== HypercubeRCCore Smoke Test ===\n\n");

    printf("--- Construction (DIM 5-12) ---\n");
    test_all_dims_construct();

    printf("\n--- Ridge prediction ---\n");
    test_ridge_prediction<5>();
    test_ridge_prediction<6>();
    test_ridge_prediction<7>();
    test_ridge_prediction<8>();

    printf("\n--- Linear prediction ---\n");
    test_linear_prediction<5>();
    test_linear_prediction<6>();

    printf("\n--- Classification ---\n");
    test_classification<6>();

    printf("\n--- Persistence ---\n");
    test_persistence<6>();

    printf("\n--- ClearStates ---\n");
    test_clear_states<6>();

    printf("\n--- TrainIncremental ---\n");
    test_train_incremental<6>();

    printf("\n--- Multi-input ---\n");
    test_multi_input<6>();

    printf("\n=== %d test(s) failed ===\n", failures);
    return failures > 0 ? 1 : 0;
}
