#pragma once

#include <cstddef>
#include <cstdint>

#include "../Reservoir.h"
#include "CNNReadout.h"

/// @file HCNNPresets.h
/// @brief Per-DIM, per-task HCNN configuration bundles.
///
/// A "preset" packages the surveyed reservoir configuration *and* the
/// tuned HCNN readout configuration for a specific benchmark at a
/// specific hypercube dimension.  Consumers get a drop-in starting point
/// that can't accidentally use the wrong reservoir seed or a stale CNN
/// default — both are wired up together.
///
/// The canonical tracker for the per-DIM tuning results, including the
/// search history that led to each preset, lives in `docs/HCNNTuning.md`.
///
/// ## Adding a new preset
///
/// 1. Run the NARMA benchmark suite until you have
///    a frozen config under `num_cnn_seeds ≥ 3` averaging.
/// 2. Add an `if constexpr (DIM == N)` block to the relevant template
///    function below, filling in both `p.reservoir` and `p.cnn`.
/// 3. Update `docs/HCNNTuning.md` with the final row and narrative.
///
/// Untuned DIMs fall through to library defaults (`CNNReadoutConfig{}`
/// with `num_layers=0` auto-sizing and the generic reservoir defaults),
/// so consumers get a functional — if unoptimized — config at any DIM.

namespace hcnn_presets {

/// Full bundle: reservoir + HCNN readout config.
struct HCNNPreset
{
    ReservoirConfig  reservoir;  ///< Includes surveyed seed when available.
    CNNReadoutConfig cnn;        ///< Tuned HCNN hyperparameters.
};

// ---------------------------------------------------------------------------
//  Per-DIM survey seeds (task-specific).
// ---------------------------------------------------------------------------
//
// The 500-seed survey is run independently per benchmark, so the winning
// seed differs between tasks even at the same DIM.  Each task-specific
// preset function picks its own surveyed seed below; there is no shared
// "SurveySeed<DIM>" — presets are the single entry point.
//
// DIM values without a surveyed seed fall through to 42 as a neutral
// fallback.  Do not hardcode 42 anywhere else — route through a preset.
//
// Note: survey seeds below were originally populated in DefaultSeed()
// functions in each benchmark header, which now delegate here.

// ---------------------------------------------------------------------------
//  NARMA-10.
// ---------------------------------------------------------------------------

/// @brief Tuned HCNN preset for the NARMA-10 benchmark.
///
/// Not yet tuned for any DIM.  All DIMs fall through to CNNReadoutConfig
/// defaults, with the surveyed reservoir seed.
template <size_t DIM>
HCNNPreset NARMA10()
{
    HCNNPreset p;

    // NARMA-specific surveyed reservoir seed (distinct from MG's).
    if      constexpr (DIM == 5) p.reservoir.seed = 2121059498467618174ULL;
    else if constexpr (DIM == 6) p.reservoir.seed = 10977843040216038077ULL;
    else if constexpr (DIM == 7) p.reservoir.seed = 6437149480297576047ULL;
    else if constexpr (DIM == 8) p.reservoir.seed = 13602423379507409791ULL;
    else if constexpr (DIM == 9) p.reservoir.seed = 10293005394405557670ULL;
    // DIM 10 has no 500-seed NARMA survey; uses the DIM 10 MG SR=0.90 winner
    // as a cross-task proxy (same seed was also the DIM 7 NARMA SR=0.90 winner,
    // making it a genuine cross-task standout).
    else if constexpr (DIM == 10) p.reservoir.seed = 6437149480297576047ULL;
    else                         p.reservoir.seed = 42ULL;  // fallback

    // CNN config: defaults until tuned per DIM.
    return p;
}

// ---------------------------------------------------------------------------
//  HRCCNN baseline — uniform first-probe architecture, DIM 5-16.
// ---------------------------------------------------------------------------

/// @brief HRCCNN baseline HCNN config — see `docs/HRCCNNBaselineConfig.md`.
///
/// Minimum-capacity first-probe architecture, uniform across every DIM:
///
///   nl=1, ch=8, ep=2000, lr_max=0.0015, bs = 1 << (DIM-1)
///
/// `ep=2000` is calibrated for **chaotic** signals (NARMA).
/// Smooth-signal tasks saturate at `ep=25`; do not reuse this config as-is
/// for smooth tasks without lowering the epoch count.
///
/// The `bs` formula holds the ~50k-gradient-update invariant constant
/// across DIMs and coincides with the DIM 5/6/7 Gold Standard `bs` values,
/// so benchmark cadence is commensurable with the tuned reference runs.
template <size_t DIM>
CNNReadoutConfig HRCCNNBaseline()
{
    CNNReadoutConfig cfg;
    cfg.num_layers    = 1;
    cfg.conv_channels = 8;
    cfg.epochs        = 2000;
    cfg.batch_size    = 1 << (DIM - 1);
    cfg.lr_max        = 0.0015f;
    // Per-DIM rank-1 CNN weight-init seeds from CnnSeedSurvey (50-seed
    // survey at DIM 5-8, 20-seed at DIM 9+ where the seed lottery is
    // cosmetic). Cross-DIM ranking shows these seeds are strong across
    // tasks, so they are used as the generic HRCCNN baseline init.
    // See diagnostics/NARMA10.md.
    if constexpr      (DIM == 5) cfg.seed = 21;
    else if constexpr (DIM == 6) cfg.seed = 34;
    else if constexpr (DIM == 7) cfg.seed = 6;
    else if constexpr (DIM == 8) cfg.seed = 2;
    else if constexpr (DIM == 9) cfg.seed = 20;
    else if constexpr (DIM == 10) cfg.seed = 2;
    return cfg;
}

}  // namespace hcnn_presets
