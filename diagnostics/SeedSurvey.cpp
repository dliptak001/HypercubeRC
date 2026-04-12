/// @file diagnostics/SeedSurvey.cpp
/// @brief Seed quality survey: run a benchmark over many random seeds.
///
/// Configure DIM, seed count, hyperparameters, and diagnostic below,
/// rebuild, and run. Reports per-seed results and distribution statistics.
///
/// Sweep modes:
///   SINGLE   — one (SR, IS) pair at a single DIM
///   SWEEP_SR — fixed IS, sweep SR values, Spearman correlation matrix
///   SWEEP_IS — fixed SR, sweep IS values, Spearman correlation matrix
///
/// Readout:
///   Ridge (default) runs the whole seed batch in parallel via OpenMP.
///   HCNN serializes the outer loop (HCNN training already saturates all
///   cores internally) and trains a CNN readout per seed — ~30-60s per
///   seed at DIM 8+, so drop SEED_COUNT accordingly.
///
/// Set SWEEP_ALL_DIMS = true to run the active sweep across DIM 5-8 (library supports 5-12).
/// Set false to run only the single DIM specified above.

#include "SeedSurvey.h"

// =====================================================================
// Configuration — change these and rebuild
// =====================================================================
static constexpr size_t DIM = 10;          // used only when SWEEP_ALL_DIMS = false
static constexpr int SEED_COUNT = 500;
static constexpr float OUTPUT_FRACTION = 0.25f;
static constexpr int DIAGNOSTIC_ID = 2;    // 0=MC, 1=MG, 2=NARMA
static constexpr int READOUT_ID = 0;       // 0=Ridge, 1=HCNN (MG/NARMA only)

enum class SweepMode { SINGLE, SWEEP_SR, SWEEP_IS };
static constexpr SweepMode MODE = SweepMode::SWEEP_SR;
static constexpr bool SWEEP_ALL_DIMS = false;

// Single mode
static constexpr float SPECTRAL_RADIUS = 0.90f;
static constexpr float INPUT_SCALING = 0.02f;

// SR sweep mode: fixed IS, vary SR
static constexpr float SR_SWEEP_IS = 0.02f;
static constexpr float SR_VALUES[] = {0.80f, 0.85f, 0.90f, 0.95f, 1.00f};

// IS sweep mode: fixed SR, vary IS
static constexpr float IS_SWEEP_SR = 0.90f;
static constexpr float IS_VALUES[] = {0.010f, 0.015f, 0.020f, 0.025f, 0.030f};

// HCNN readout config — matches BenchmarkSuite --hcnn defaults so HCNN
// survey results are directly comparable to the benchmark numbers.
// Drop epochs if you want a faster scan on a smoother task.
static CNNReadoutConfig HcnnSurveyConfig()
{
    CNNReadoutConfig cfg;
    cfg.epochs = 300;
    cfg.batch_size = 128;
    cfg.lr_max = 0.003f;
    return cfg;
}

// =====================================================================

template <size_t D>
void RunOne()
{
    using Survey = SeedSurvey<D>;
    constexpr auto diagnostic = static_cast<typename Survey::Diagnostic>(DIAGNOSTIC_ID);
    constexpr auto readout    = static_cast<typename Survey::Readout>(READOUT_ID);
    const CNNReadoutConfig hcnn_cfg = HcnnSurveyConfig();

    if constexpr (MODE == SweepMode::SWEEP_SR)
    {
        std::vector<float> srs(std::begin(SR_VALUES), std::end(SR_VALUES));
        Survey::RunCorrelation(SEED_COUNT, srs, SR_SWEEP_IS, diagnostic,
                               OUTPUT_FRACTION, readout, hcnn_cfg);
    }
    else if constexpr (MODE == SweepMode::SWEEP_IS)
    {
        std::vector<float> iss(std::begin(IS_VALUES), std::end(IS_VALUES));
        Survey::RunCorrelationIS(SEED_COUNT, IS_SWEEP_SR, iss, diagnostic,
                                 OUTPUT_FRACTION, readout, hcnn_cfg);
    }
    else
    {
        Survey survey(SEED_COUNT, SPECTRAL_RADIUS, INPUT_SCALING, diagnostic,
                      OUTPUT_FRACTION, readout, hcnn_cfg);
        survey.Run();
    }
}

int main()
{
    if constexpr (SWEEP_ALL_DIMS)
    {
        RunOne<5>();
        RunOne<6>();
        RunOne<7>();
        RunOne<8>();
    }
    else
    {
        RunOne<DIM>();
    }
    return 0;
}
