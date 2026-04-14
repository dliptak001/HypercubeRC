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
/// DIM 5 **GOLD STANDARD** (frozen 2026-04-13, runs 15-20):
///   nl=1, ch=16, head=FLAT, ep=2000, bs=16, **lr_max=0.0015**
///   → averaged NRMSE 0.003169 (10 CNN inits on the survey reservoir),
///     **-49% vs Ridge raw (0.00616)**,
///     **-28% vs Ridge translated (0.00438)**.
///
///   Training dynamics at this backbone favor a lower learning rate than
///   the old nl=3/ch=32/GAP preset used: runs 19-20 traced a clean bowl
///   from lr∈{0.0015, 0.002, 0.0025, 0.003, 0.0035} with the minimum
///   cleanly at 0.0015.  The coarser decade sweep in run 18 had missed
///   it by spacing.
///
///   The architecture itself (nl=1 FLATTEN) was pinned in runs 15-17:
///   GAP and FLATTEN have opposite backbone preferences on a hypercube
///   reservoir — GAP wants deep+wide, FLATTEN wants minimum-depth +
///   fan-in-sized channels.  For FLATTEN a second conv+pool destroys
///   per-vertex structure that the readout needs.  See `docs/HCNNTuning.md`
///   runs 15-20 for the full narrative.
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
        p.cnn.num_layers    = 1;
        p.cnn.conv_channels = 16;
        p.cnn.readout_type  = HCNNReadoutType::FLATTEN;
        p.cnn.epochs        = 2000;
        p.cnn.batch_size    = 16;
        p.cnn.lr_max        = 0.0015f;
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
