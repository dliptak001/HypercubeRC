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
/// MG and NARMA use RidgeRegression; MC uses LinearReadout (standard for
/// memory capacity). All use per-DIM optimized spectral radius and input
/// scaling, and collect = 18*N training samples (5x oversampling for the
/// 2.5N feature count).

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <omp.h>
#include "ESN.h"
#include "TranslationLayer.h"
#include "readout/LinearReadout.h"
#include "diagnostics/MackeyGlass.h"
#include "diagnostics/NARMA10.h"

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
// MG benchmark — delegates to MackeyGlass<DIM> diagnostic class.
// ---------------------------------------------------------------------------
template <size_t DIM>
static void RunMG(size_t horizon)
{
    MackeyGlass<DIM> mg(horizon, ReadoutType::Ridge);
    auto r = mg.Run();
    std::cout << "  " << std::setw(3) << DIM
        << "  | " << std::setw(5) << (1ULL << DIM)
        << " | " << std::fixed << std::setprecision(4) << std::setw(7) << r.nrmse_raw
        << " | " << std::setprecision(4) << std::setw(7) << r.nrmse_full
        << " (" << std::showpos << std::setprecision(1) << std::setw(5) << r.pct_change
        << "%" << std::noshowpos << ")\n";
}

// ---------------------------------------------------------------------------
// NARMA-10 benchmark — delegates to NARMA10<DIM> diagnostic class.
// ---------------------------------------------------------------------------
template <size_t DIM>
static void RunNARMA()
{
    NARMA10<DIM> narma(ReadoutType::Ridge);
    auto r = narma.Run();
    std::cout << "  " << std::setw(3) << DIM
        << "  | " << std::setw(5) << (1ULL << DIM)
        << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r.nrmse_raw
        << " | " << std::setprecision(3) << std::setw(7) << r.nrmse_full
        << " (" << std::showpos << std::setprecision(1) << std::setw(5) << r.pct_change
        << "%" << std::noshowpos << ")\n";
}

// ---------------------------------------------------------------------------
int main()
{
    const std::vector<uint64_t> seeds = {42, 1042, 2042};

    std::cout << "=== HypercubeRC Benchmark Suite ===\n\n";
    std::cout << "HypercubeRC is a reservoir computer whose N neurons are arranged on\n";
    std::cout << "a Boolean hypercube -- a DIM-dimensional graph where each vertex is\n";
    std::cout << "addressed by a DIM-bit binary index, and two vertices are neighbors\n";
    std::cout << "if their indices differ by one bit (computed by XOR, no adjacency\n";
    std::cout << "list needed). Input and output are continuous scalar values.\n\n";
    std::cout << "The pipeline: drive the reservoir with a scalar input signal, collect\n";
    std::cout << "the N-dimensional state at each step, then train a linear readout to\n";
    std::cout << "map those states to the target. The reservoir's weights are fixed --\n";
    std::cout << "only the readout is learned. All results below are 3-seed averages.\n\n";
    std::cout << "  DIM  -- hypercube dimension; the reservoir has N = 2^DIM neurons\n";
    std::cout << "  raw  -- readout uses N raw reservoir states\n";
    std::cout << "  full -- readout uses 2.5N features after translation (x, x^2, x*x')\n";
    std::cout << "          the extra nonlinear terms help the linear readout decode\n";
    std::cout << "          information that tanh folds into the state vector\n\n";

    std::cout << "--- Memory Capacity (lags 1-50) ---\n";
    std::cout << "How many past inputs can the reservoir reconstruct? MC = sum of R^2\n";
    std::cout << "over lags 1-50. Higher is better. Theoretical max = N.\n\n";
    std::cout << "  DIM |     N |    MC\n";
    std::cout << "  ----+-------+------\n";
    RunMC<5>(seeds);
    RunMC<6>(seeds);
    RunMC<7>(seeds);
    RunMC<8>(seeds);
    //RunMC<9>(seeds); RunMC<10>(seeds);

    std::cout << "\n--- Mackey-Glass h=1 (NRMSE, lower is better) ---\n";
    std::cout << "One-step prediction of a chaotic time series. Tests how well the\n";
    std::cout << "reservoir tracks complex, deterministic dynamics.\n\n";
    std::cout << "  DIM |     N |    raw  |   full translation\n";
    std::cout << "  ----+-------+---------+-----------------\n";
    RunMG<5>(1);
    RunMG<6>(1);
    RunMG<7>(1);
    RunMG<8>(1);
    //RunMG<9>(1); RunMG<10>(1);

    std::cout << "\n--- NARMA-10 (NRMSE, lower is better) ---\n";
    std::cout << "Nonlinear autoregressive benchmark requiring both memory (10-step\n";
    std::cout << "history) and nonlinear computation (product terms). This is where\n";
    std::cout << "the translation layer has the biggest impact.\n\n";
    std::cout << "  DIM |     N |    raw  |   full translation\n";
    std::cout << "  ----+-------+---------+-----------------\n";
    RunNARMA<5>();
    RunNARMA<6>();
    RunNARMA<7>();
    RunNARMA<8>();
    //RunNARMA<9>(); RunNARMA<10>();

    return 0;
}
