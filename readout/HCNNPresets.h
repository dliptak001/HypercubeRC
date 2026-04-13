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
/// 1. Run `OptimizeHRCCNNForMG` (or the NARMA counterpart) until you have
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
// Note: the MackeyGlass and NARMA-10 survey seeds below were originally
// populated in MackeyGlass.h / NARMA10.h `DefaultSeed()` functions,
// which now delegate here.

// ---------------------------------------------------------------------------
//  Mackey-Glass (horizon 1).
// ---------------------------------------------------------------------------

/// @brief Tuned HCNN preset for the Mackey-Glass chaotic benchmark.
///
/// DIM 5 (frozen 2026-04-13): nl=3, ch=32, ep=2000, bs=16, lr_max=0.003
///   → averaged NRMSE 0.00490 (3 CNN inits on the survey reservoir),
///     -21% vs Ridge raw (0.00616), +12% vs Ridge translated (0.00438).
///   Note: nl=3 exceeds CNNReadout's auto-rule cap `min(DIM-3,4)`=2.
///   The auto-rule's assert is off-by-one — real HCNNConv constraint is
///   `nl ≤ DIM-2`, and depth past the auto-rule is a real improvement
///   at DIM=5.  See `docs/HCNNTuning.md` for the full tuning history.
template <size_t DIM>
HCNNPreset MackeyGlass()
{
    HCNNPreset p;

    // MG-specific surveyed reservoir seed.
    if      constexpr (DIM == 5) p.reservoir.seed = 11822067163148543833ULL;
    else if constexpr (DIM == 6) p.reservoir.seed = 11459651989651327597ULL;
    else if constexpr (DIM == 7) p.reservoir.seed = 10741866950647888161ULL;
    else if constexpr (DIM == 8) p.reservoir.seed = 2121059498467618174ULL;
    else                         p.reservoir.seed = 42ULL;  // fallback

    if constexpr (DIM == 5)
    {
        p.cnn.num_layers    = 3;
        p.cnn.conv_channels = 32;
        p.cnn.epochs        = 2000;
        p.cnn.batch_size    = 16;
        p.cnn.lr_max        = 0.003f;
        // lr_min_frac, weight_decay, seed, num_outputs, task: defaults.
    }
    // DIM 6+: untuned.  Falls through to CNNReadoutConfig defaults.

    return p;
}

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
    else                         p.reservoir.seed = 42ULL;  // fallback

    // CNN config: defaults until tuned per DIM.
    return p;
}

}  // namespace hcnn_presets
