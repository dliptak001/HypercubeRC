#pragma once

#include "Reservoir.h"

/// Feature mode selector for per-DIM default resolution.
/// Raw: optimized for N-dim raw readout.
/// Translation: optimized for 2.5N-dim readout via TranslationLayer.
enum class FeatureMode { Raw, Translation };

/// @brief Per-DIM optimized defaults for reservoir parameters.
///
/// Application-level policy — not part of the Reservoir kernel.
/// Diagnostics, examples, and sweep tools use this to build a
/// ReservoirConfig with per-DIM tuned values. Callers who already
/// have concrete values can bypass this entirely.
///
/// Two sets of defaults:
///   - **Raw**: optimized for N-dim raw readout (no translation layer).
///   - **Translation**: optimized for 2.5N-dim readout via TranslationLayer.
///
/// All values jointly optimized on MG h=1 + NARMA-10 + Memory Capacity.
template <size_t DIM>
struct ReservoirDefaults
{
    static_assert(DIM >= 4 && DIM <= 10, "DIM must be in 4 <= DIM <= 10");

    // ----- Raw-feature defaults -----
    // Jointly optimized on MG h=1 + NARMA-10 + MC, raw N-dim readout.
    // Balanced across all tasks, not single-task optimal.

    static constexpr float RawSpectralRadius()
    {
        if constexpr (DIM == 4) return 0.95f;
        if constexpr (DIM == 5) return 0.80f;
        if constexpr (DIM == 6) return 0.90f;
        if constexpr (DIM == 7) return 0.88f;
        if constexpr (DIM == 8) return 0.88f;
        if constexpr (DIM == 9) return 0.88f;   // extrapolated from DIM 8, not sweep-verified
        if constexpr (DIM == 10) return 0.88f;  // extrapolated from DIM 8, not sweep-verified
        return 0.88f;
    }

    static constexpr float RawInputScaling()
    {
        if constexpr (DIM == 4) return 0.05f;
        if constexpr (DIM == 5) return 0.10f;
        if constexpr (DIM == 6) return 0.05f;
        if constexpr (DIM == 7) return 0.03f;
        if constexpr (DIM == 8) return 0.02f;
        if constexpr (DIM == 9) return 0.02f;   // extrapolated from DIM 8, not sweep-verified
        if constexpr (DIM == 10) return 0.02f;  // extrapolated from DIM 8, not sweep-verified
        return 0.02f;
    }

    // ----- Translation-layer defaults -----
    // Jointly optimized on MG h=1 + NARMA-10 + MC, translation 2.5N readout.
    // Balanced across all tasks. Very low input scaling (0.02-0.04) —
    // translation features amplify dynamics, so less drive is needed.

    static constexpr float TranslationSpectralRadius()
    {
        if constexpr (DIM == 4) return 0.88f;
        if constexpr (DIM == 5) return 0.80f;
        if constexpr (DIM == 6) return 0.92f;
        if constexpr (DIM == 7) return 0.92f;
        if constexpr (DIM == 8) return 0.95f;
        if constexpr (DIM == 9) return 0.95f;   // extrapolated from DIM 8, not sweep-verified
        if constexpr (DIM == 10) return 0.95f;  // extrapolated from DIM 8, not sweep-verified
        return 0.95f;
    }

    static constexpr float TranslationInputScaling()
    {
        if constexpr (DIM == 4) return 0.02f;
        if constexpr (DIM == 5) return 0.04f;
        if constexpr (DIM == 6) return 0.02f;
        if constexpr (DIM == 7) return 0.04f;
        if constexpr (DIM == 8) return 0.02f;
        if constexpr (DIM == 9) return 0.02f;   // extrapolated from DIM 8, not sweep-verified
        if constexpr (DIM == 10) return 0.02f;  // extrapolated from DIM 8, not sweep-verified
        return 0.02f;
    }

    /// Resolve the default spectral radius for a given feature mode.
    static constexpr float DefaultSpectralRadius(FeatureMode mode = FeatureMode::Raw)
    {
        return (mode == FeatureMode::Translation) ? TranslationSpectralRadius() : RawSpectralRadius();
    }

    /// Resolve the default input scaling for a given feature mode.
    static constexpr float DefaultInputScaling(FeatureMode mode = FeatureMode::Raw)
    {
        return (mode == FeatureMode::Translation) ? TranslationInputScaling() : RawInputScaling();
    }

    /// Build a fully resolved ReservoirConfig for the given feature mode.
    static ReservoirConfig MakeConfig(uint64_t seed, FeatureMode mode = FeatureMode::Raw)
    {
        ReservoirConfig cfg;
        cfg.seed = seed;
        cfg.spectral_radius = DefaultSpectralRadius(mode);
        cfg.block_scaling = {DefaultInputScaling(mode)};
        return cfg;
    }
};
