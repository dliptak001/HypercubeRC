/// @file main.cpp
/// @brief HypercubeRC benchmark suite.
///
/// Runs three standard reservoir computing benchmarks across DIM 5-10,
/// each averaged over 3 seeds {42, 1042, 2042}:
///
///   - **Memory Capacity (MC):** Sum of R² over lags 1-50. Measures how much
///     past input the reservoir can reconstruct via a linear readout.
///
///   - **Mackey-Glass h=1 (MG):** One-step-ahead prediction of a chaotic time
///     series (tau=17 delay equation). Reports NRMSE with and without the
///     full translation layer. Standard ESN range: 0.01-0.05.
///
///   - **NARMA-10:** Nonlinear autoregressive benchmark requiring both memory
///     (10-step history) and nonlinear computation (product terms). Reports
///     NRMSE with and without translation. Standard ESN range: 0.2-0.4.
///
/// All benchmarks use LinearReadout, full translation layer (2.5N features),
/// per-DIM optimized spectral radius and input scaling, and collect = 18*N
/// training samples (5x oversampling for the 2.5N feature count).

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include "ESN.h"
#include "TranslationLayer.h"
#include "SignalGenerators.h"

static std::pair<double, double> EvalLinear(const float* features, const float* targets,
                                            size_t train, size_t test, size_t nf)
{
    LinearReadout lr;
    lr.Train(features, targets, train, nf);
    double r2 = lr.R2(features + train * nf, targets + train, test);
    std::vector<float> pred(test);
    for (size_t s = 0; s < test; ++s)
        pred[s] = lr.PredictRaw(features + (train + s) * nf);
    double nrmse = ComputeNRMSE(pred.data(), targets + train, test);
    return {nrmse, r2};
}

// ---------------------------------------------------------------------------
// Sizing: collect = 18*N ensures 5x oversampling for 2.5N translation features.
// ---------------------------------------------------------------------------
template <size_t DIM>
constexpr size_t Warmup() { return ((1ULL << DIM) < 256) ? 200 : 500; }

template <size_t DIM>
constexpr size_t Collect() { return 18 * (1ULL << DIM); }

// ---------------------------------------------------------------------------
// MC benchmark (LinearReadout, raw features only)
// ---------------------------------------------------------------------------
template <size_t DIM>
static void RunMC(const std::vector<uint64_t>& seeds, size_t max_lag = 50)
{
    constexpr size_t N = 1ULL << DIM;
    double s_mc = 0;

    for (uint64_t seed : seeds)
    {
        std::mt19937_64 rng(seed + 99);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        size_t total = Warmup<DIM>() + Collect<DIM>();
        std::vector<float> inputs(total);
        for (size_t i = 0; i < total; ++i)
            inputs[i] = static_cast<float>(dist(rng));

        ESN<DIM> esn(seed, ReadoutType::Linear);
        esn.Warmup(inputs.data(), Warmup<DIM>());
        esn.Run(inputs.data() + Warmup<DIM>(), Collect<DIM>());

        const float* st = esn.States();
        double mc = 0.0;
        size_t num_lags = std::min(max_lag, Collect<DIM>() - 1);

#pragma omp parallel for reduction(+:mc) schedule(dynamic)
        for (size_t lag = 1; lag <= num_lags; ++lag)
        {
            size_t valid = Collect<DIM>() - lag;
            std::vector<float> tgt(valid);
            for (size_t t = 0; t < valid; ++t)
                tgt[t] = inputs[Warmup<DIM>() + t];

            const float* vs = st + lag * N;
            size_t tr = static_cast<size_t>(valid * 0.7);
            size_t te = valid - tr;
            if (tr == 0 || te == 0) continue;

            LinearReadout lr;
            lr.Train(vs, tgt.data(), tr, N);
            double r2 = lr.R2(vs + tr * N, tgt.data() + tr, te);
            if (r2 > 0.0) mc += r2;
        }
        s_mc += mc;
    }

    double mc = s_mc / static_cast<double>(seeds.size());
    std::cout << "  " << std::setw(3) << DIM
        << "  | " << std::setw(5) << N
        << " | " << std::fixed << std::setprecision(1) << std::setw(5) << mc << "\n";
}

// ---------------------------------------------------------------------------
// MG benchmark (LinearReadout, raw vs full translation)
// ---------------------------------------------------------------------------
template <size_t DIM>
static void RunMG(const std::vector<uint64_t>& seeds, size_t horizon)
{
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t FULL = TranslationFeatureCount<DIM>();
    size_t collect = Collect<DIM>() - horizon;

    double s_nr = 0, s_fnr = 0;

    for (uint64_t seed : seeds)
    {
        auto series = GenerateMackeyGlass(Warmup<DIM>() + Collect<DIM>() + 20);
        Normalize(series);

        std::vector<float> tgt(collect);
        for (size_t t = 0; t < collect; ++t) tgt[t] = series[Warmup<DIM>() + t + horizon];
        size_t tr = static_cast<size_t>(collect * 0.7), te = collect - tr;

        // Raw features — raw-optimized defaults
        {
            ESN<DIM> esn(seed, ReadoutType::Linear);
            esn.Warmup(series.data(), Warmup<DIM>());
            esn.Run(series.data() + Warmup<DIM>(), collect);
            auto [nr, _r] = EvalLinear(esn.States(), tgt.data(), tr, te, N);
            s_nr += nr;
        }

        // Translation features — translation-optimized defaults
        {
            ESN<DIM> esn(seed, ReadoutType::Linear, FeatureMode::Translation);
            esn.Warmup(series.data(), Warmup<DIM>());
            esn.Run(series.data() + Warmup<DIM>(), collect);
            auto full = TranslationTransform<DIM>(esn.States(), collect);
            auto [fnr, _f] = EvalLinear(full.data(), tgt.data(), tr, te, FULL);
            s_fnr += fnr;
        }
    }

    double n = static_cast<double>(seeds.size());
    double nr = s_nr / n, fnr = s_fnr / n;
    double pct = (nr > 1e-12) ? 100.0 * (fnr - nr) / nr : 0.0;
    std::cout << "  " << std::setw(3) << DIM
        << "  | " << std::setw(5) << (1ULL << DIM)
        << " | " << std::fixed << std::setprecision(4) << std::setw(7) << nr
        << " | " << std::setprecision(4) << std::setw(7) << fnr
        << " (" << std::showpos << std::setprecision(1) << std::setw(5) << pct
        << "%" << std::noshowpos << ")\n";
}

// ---------------------------------------------------------------------------
// NARMA-10 benchmark (LinearReadout, raw vs full translation)
// ---------------------------------------------------------------------------
template <size_t DIM>
static void RunNARMA(const std::vector<uint64_t>& seeds)
{
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t FULL = TranslationFeatureCount<DIM>();
    size_t collect = Collect<DIM>();

    double s_nr = 0, s_fnr = 0;

    for (uint64_t seed : seeds)
    {
        auto [u, y] = GenerateNARMA10(seed + 99, Warmup<DIM>() + collect);

        std::vector<float> ri(Warmup<DIM>() + collect);
        for (size_t t = 0; t < ri.size(); ++t) ri[t] = u[t] * 4.0f - 1.0f;

        std::vector<float> tgt(collect);
        for (size_t t = 0; t < collect; ++t) tgt[t] = y[Warmup<DIM>() + t];
        size_t tr = static_cast<size_t>(collect * 0.7), te = collect - tr;

        // Raw features — raw-optimized defaults
        {
            ESN<DIM> esn(seed, ReadoutType::Linear);
            esn.Warmup(ri.data(), Warmup<DIM>());
            esn.Run(ri.data() + Warmup<DIM>(), collect);
            auto [nr, _r] = EvalLinear(esn.States(), tgt.data(), tr, te, N);
            s_nr += nr;
        }

        // Translation features — translation-optimized defaults
        {
            ESN<DIM> esn(seed, ReadoutType::Linear, FeatureMode::Translation);
            esn.Warmup(ri.data(), Warmup<DIM>());
            esn.Run(ri.data() + Warmup<DIM>(), collect);
            auto full = TranslationTransform<DIM>(esn.States(), collect);
            auto [fnr, _f] = EvalLinear(full.data(), tgt.data(), tr, te, FULL);
            s_fnr += fnr;
        }
    }

    double n = static_cast<double>(seeds.size());
    double nr = s_nr / n, fnr = s_fnr / n;
    double pct = (nr > 1e-12) ? 100.0 * (fnr - nr) / nr : 0.0;
    std::cout << "  " << std::setw(3) << DIM
        << "  | " << std::setw(5) << (1ULL << DIM)
        << " | " << std::fixed << std::setprecision(3) << std::setw(7) << nr
        << " | " << std::setprecision(3) << std::setw(7) << fnr
        << " (" << std::showpos << std::setprecision(1) << std::setw(5) << pct
        << "%" << std::noshowpos << ")\n";
}

// ---------------------------------------------------------------------------
int main()
{
    const std::vector<uint64_t> seeds = {42, 1042, 2042};

    std::cout << "=== Full Benchmark Suite (LinearReadout, 3-seed avg) ===\n\n";

    std::cout << "--- MC (lags 1-50) ---\n";
    std::cout << "  DIM |     N |    MC\n";
    std::cout << "  ----+-------+------\n";
    RunMC<5>(seeds);
    RunMC<6>(seeds);
    RunMC<7>(seeds);
    RunMC<8>(seeds);
    //RunMC<9>(seeds); RunMC<10>(seeds);

    std::cout << "\n--- MG h=1 (lower is better) ---\n";
    std::cout << "  DIM |     N |    raw  |   full translation\n";
    std::cout << "  ----+-------+---------+-----------------\n";
    RunMG<5>(seeds, 1);
    RunMG<6>(seeds, 1);
    RunMG<7>(seeds, 1);
    RunMG<8>(seeds, 1);
    //RunMG<9>(seeds, 1); RunMG<10>(seeds, 1);

    std::cout << "\n--- NARMA-10 (lower is better) ---\n";
    std::cout << "  DIM |     N |    raw  |   full translation\n";
    std::cout << "  ----+-------+---------+-----------------\n";
    RunNARMA<5>(seeds);
    RunNARMA<6>(seeds);
    RunNARMA<7>(seeds);
    RunNARMA<8>(seeds);
    //RunNARMA<9>(seeds); RunNARMA<10>(seeds);

    return 0;
}
