/// @file sweeps/StandaloneESNSweep.cpp
/// @brief Grid sweep: SR x input_scaling for standalone ESN.
///
/// Configure DIM, USE_TRANSLATION, and sweep ranges below, rebuild, and run.
/// 3-seed average. Runs MG h=1, NARMA-10, and MC.

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "../ESN.h"
#include "../TranslationLayer.h"
#include "../SignalGenerators.h"
#include "../readout/LinearReadout.h"
#include "../readout/RidgeRegression.h"

// =====================================================================
// Configuration — change these and rebuild
// =====================================================================
static constexpr size_t DIM = 8;
static constexpr bool USE_TRANSLATION = true;
static constexpr ReadoutType READOUT = ReadoutType::Linear;

static constexpr size_t N = 1ULL << DIM;
static constexpr size_t NF = USE_TRANSLATION ? TranslationFeatureCount<DIM>() : N;
static constexpr size_t WARMUP = (DIM >= 8) ? 500 : 200;
static constexpr size_t COLLECT = 18 * N;
static constexpr size_t MAX_LAG = 50;

static const std::vector<uint64_t> SEEDS = {42, 1042, 2042};

// Sweep grid — adjust per experiment
static const std::vector<float> SR_VALUES  = {0.93f, 0.94f, 0.95f, 0.96f, 0.97f};
static const std::vector<float> INP_VALUES = {0.02f, 0.04f, 0.06f, 0.10f, 0.15f};

// =====================================================================
// Feature extraction helper
// =====================================================================
static std::vector<float> GetFeatures(const float* states, size_t num_samples)
{
    if constexpr (USE_TRANSLATION)
        return TranslationTransform<DIM>(states, num_samples);
    else
    {
        // Return a copy so the interface is uniform
        return std::vector<float>(states, states + num_samples * N);
    }
}

// =====================================================================
// Eval helper
// =====================================================================
static double EvalNRMSE(const float* features, const float* targets,
                        size_t train, size_t test)
{
    auto eval = [&](auto& readout)
    {
        readout.Train(features, targets, train, NF);
        return ComputeNRMSE(readout, features + train * NF, targets + train, test, NF);
    };

    if constexpr (READOUT == ReadoutType::Ridge)
    { RidgeRegression r; return eval(r); }
    else
    { LinearReadout r; return eval(r); }
}

// =====================================================================
// Benchmark runners
// =====================================================================
static double RunMG(float sr, float inp)
{
    constexpr size_t collect = COLLECT - 1;
    double sum = 0.0;
    for (uint64_t seed : SEEDS)
    {
        auto series = GenerateMackeyGlass(WARMUP + COLLECT + 20);
        Normalize(series);

        std::vector<float> tgt(collect);
        for (size_t t = 0; t < collect; ++t)
            tgt[t] = series[WARMUP + t + 1];

        size_t tr = static_cast<size_t>(collect * 0.7);
        size_t te = collect - tr;

        // SR and input scaling are explicitly controlled; FeatureMode selects
        // the matching default for any params NOT overridden (none here).
        constexpr auto MODE = USE_TRANSLATION ? FeatureMode::Translation : FeatureMode::Raw;
        float bs[1] = {inp};
        ESN<DIM> esn(seed, READOUT, MODE, 1.0f, sr, bs);
        esn.Warmup(series.data(), WARMUP);
        esn.Run(series.data() + WARMUP, collect);

        auto feat = GetFeatures(esn.States(), collect);
        sum += EvalNRMSE(feat.data(), tgt.data(), tr, te);
    }
    return sum / static_cast<double>(SEEDS.size());
}

static double RunNARMA(float sr, float inp)
{
    double sum = 0.0;
    for (uint64_t seed : SEEDS)
    {
        auto [u, y] = GenerateNARMA10(seed + 99, WARMUP + COLLECT);

        std::vector<float> ri(WARMUP + COLLECT);
        for (size_t t = 0; t < ri.size(); ++t)
            ri[t] = u[t] * 4.0f - 1.0f;

        std::vector<float> tgt(COLLECT);
        for (size_t t = 0; t < COLLECT; ++t)
            tgt[t] = y[WARMUP + t];

        size_t tr = static_cast<size_t>(COLLECT * 0.7);
        size_t te = COLLECT - tr;

        constexpr auto MODE = USE_TRANSLATION ? FeatureMode::Translation : FeatureMode::Raw;
        float bs[1] = {inp};
        ESN<DIM> esn(seed, READOUT, MODE, 1.0f, sr, bs);
        esn.Warmup(ri.data(), WARMUP);
        esn.Run(ri.data() + WARMUP, COLLECT);

        auto feat = GetFeatures(esn.States(), COLLECT);
        sum += EvalNRMSE(feat.data(), tgt.data(), tr, te);
    }
    return sum / static_cast<double>(SEEDS.size());
}

static double RunMC(float sr, float inp)
{
    double sum = 0.0;
    for (uint64_t seed : SEEDS)
    {
        std::mt19937_64 rng(seed + 99);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<float> inputs(WARMUP + COLLECT);
        for (size_t i = 0; i < inputs.size(); ++i)
            inputs[i] = static_cast<float>(dist(rng));

        constexpr auto MODE = USE_TRANSLATION ? FeatureMode::Translation : FeatureMode::Raw;
        float bs[1] = {inp};
        ESN<DIM> esn(seed, READOUT, MODE, 1.0f, sr, bs);
        esn.Warmup(inputs.data(), WARMUP);
        esn.Run(inputs.data() + WARMUP, COLLECT);

        auto feat = GetFeatures(esn.States(), COLLECT);

        double mc = 0.0;
        for (size_t lag = 1; lag <= MAX_LAG && lag < COLLECT; ++lag)
        {
            size_t valid = COLLECT - lag;
            size_t tr = static_cast<size_t>(valid * 0.7);
            size_t te = valid - tr;
            if (tr == 0 || te == 0) continue;

            std::vector<float> tgt(valid);
            for (size_t t = 0; t < valid; ++t)
                tgt[t] = inputs[WARMUP + t];

            const float* lagged = feat.data() + lag * NF;

            auto eval_r2 = [&](auto& readout)
            {
                readout.Train(lagged, tgt.data(), tr, NF);
                return readout.R2(lagged + tr * NF, tgt.data() + tr, te);
            };

            double r2;
            if constexpr (READOUT == ReadoutType::Ridge)
            { RidgeRegression r; r2 = eval_r2(r); }
            else
            { LinearReadout r; r2 = eval_r2(r); }
            if (r2 > 0.0) mc += r2;
        }
        sum += mc;
    }
    return sum / static_cast<double>(SEEDS.size());
}

// =====================================================================
// Main — grid sweep with formatted output
// =====================================================================
int main()
{
    size_t total = SR_VALUES.size() * INP_VALUES.size();
    const char* mode = USE_TRANSLATION ? "translation (2.5N)" : "raw (N)";
    const char* rn = (READOUT == ReadoutType::Ridge) ? "Ridge" : "Linear";

    std::cout << "=== Standalone ESN Sweep: DIM=" << DIM << " N=" << N
              << " NF=" << NF << " " << mode << " " << rn
              << " (" << total << " configs, 3-seed avg) ===\n\n";

    std::cout << "    SR |  inp |    MG h1 | NARMA-10 |    MC\n";
    std::cout << "  -----+------+----------+----------+------\n";

    for (float sr : SR_VALUES)
    {
        for (float inp : INP_VALUES)
        {
            double mg    = RunMG(sr, inp);
            double narma = RunNARMA(sr, inp);
            double mc    = RunMC(sr, inp);

            std::cout << std::fixed << std::setprecision(2)
                      << "  " << std::setw(4) << sr
                      << " | " << std::setw(4) << inp
                      << " | " << std::setprecision(5) << std::setw(8) << mg
                      << " | " << std::setprecision(4) << std::setw(8) << narma
                      << " | " << std::setprecision(2) << std::setw(5) << mc
                      << "\n";
        }
    }

    return 0;
}
