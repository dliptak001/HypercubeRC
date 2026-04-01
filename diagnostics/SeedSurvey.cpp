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
/// Set SWEEP_ALL_DIMS = true to run the active sweep across DIM 5-8.
/// Set false to run only the single DIM specified above.

#include "SeedSurvey.h"

// =====================================================================
// Configuration — change these and rebuild
// =====================================================================
static constexpr size_t DIM = 5;           // used only when SWEEP_ALL_DIMS = false
static constexpr int SEED_COUNT = 500;
static constexpr float OUTPUT_FRACTION = 1.0f;
static constexpr int DIAGNOSTIC_ID = 2;    // 0=MC, 1=MG, 2=NARMA

enum class SweepMode { SINGLE, SWEEP_SR, SWEEP_IS };
static constexpr SweepMode MODE = SweepMode::SWEEP_SR;
static constexpr bool SWEEP_ALL_DIMS = true;

// Single mode
static constexpr float SPECTRAL_RADIUS = 0.90f;
static constexpr float INPUT_SCALING = 0.02f;

// SR sweep mode: fixed IS, vary SR
static constexpr float SR_SWEEP_IS = 0.02f;
static constexpr float SR_VALUES[] = {0.80f, 0.85f, 0.90f, 0.95f, 1.00f};

// IS sweep mode: fixed SR, vary IS
static constexpr float IS_SWEEP_SR = 0.90f;
static constexpr float IS_VALUES[] = {0.010f, 0.015f, 0.020f, 0.025f, 0.030f};

// =====================================================================

template <size_t D>
void RunOne()
{
    using Survey = SeedSurvey<D>;
    constexpr auto diagnostic = static_cast<typename Survey::Diagnostic>(DIAGNOSTIC_ID);

    if constexpr (MODE == SweepMode::SWEEP_SR)
    {
        std::vector<float> srs(std::begin(SR_VALUES), std::end(SR_VALUES));
        Survey::RunCorrelation(SEED_COUNT, srs, SR_SWEEP_IS, diagnostic, OUTPUT_FRACTION);
    }
    else if constexpr (MODE == SweepMode::SWEEP_IS)
    {
        std::vector<float> iss(std::begin(IS_VALUES), std::end(IS_VALUES));
        Survey::RunCorrelationIS(SEED_COUNT, IS_SWEEP_SR, iss, diagnostic, OUTPUT_FRACTION);
    }
    else
    {
        Survey survey(SEED_COUNT, SPECTRAL_RADIUS, INPUT_SCALING, diagnostic, OUTPUT_FRACTION);
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
