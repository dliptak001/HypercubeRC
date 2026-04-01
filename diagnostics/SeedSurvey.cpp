/// @file diagnostics/SeedSurvey.cpp
/// @brief Seed quality survey: run a benchmark over many random seeds.
///
/// Configure DIM, seed count, hyperparameters, and diagnostic below,
/// rebuild, and run. Reports per-seed results and distribution statistics.
///
/// Set CORRELATION_MODE = true to run at multiple SR values and compute
/// pairwise Spearman rank correlation.

#include "SeedSurvey.h"

// =====================================================================
// Configuration — change these and rebuild
// =====================================================================
static constexpr size_t DIM = 8;
static constexpr int SEED_COUNT = 500;
static constexpr float INPUT_SCALING = 0.02f;
static constexpr float OUTPUT_FRACTION = 1.0f;
static constexpr auto DIAGNOSTIC = SeedSurvey<DIM>::Diagnostic::Mackey_Glass;

// Single-SR mode
static constexpr float SPECTRAL_RADIUS = 0.90f;

// Correlation mode: run at multiple SR values, compute Spearman matrix
static constexpr bool CORRELATION_MODE = true;
static constexpr float SR_VALUES[] = {0.80f, 0.85f, 0.90f, 0.95f, 1.00f};

int main()
{
    if constexpr (CORRELATION_MODE)
    {
        std::vector<float> srs(std::begin(SR_VALUES), std::end(SR_VALUES));
        SeedSurvey<DIM>::RunCorrelation(SEED_COUNT, srs, INPUT_SCALING,
                                        DIAGNOSTIC, OUTPUT_FRACTION);
    }
    else
    {
        SeedSurvey<DIM> survey(SEED_COUNT, SPECTRAL_RADIUS, INPUT_SCALING,
                               DIAGNOSTIC, OUTPUT_FRACTION);
        survey.Run();
    }
    return 0;
}
