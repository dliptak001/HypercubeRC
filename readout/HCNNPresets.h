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
/// DIM 7 **GOLD STANDARD** (frozen 2026-04-14, runs 27-30):
///   nl=2, ch=24, head=FLAT, ep=2000, bs=64, **lr_max=0.0015**
///   → averaged NRMSE 0.001494 (10 CNN inits on the survey reservoir),
///     **-65.3% vs Ridge raw (0.00431)**,
///     **-52.8% vs Ridge translated (0.00316)**.
///   First DIM where the "nl=1 + ch=16" small-N sweet spot breaks;
///   deeper and wider both win.  See DIM 7 block comment for details.
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
    else if constexpr (DIM == 6)
    {
        // DIM 6 GOLD STANDARD (frozen 2026-04-13, runs 21-26): same
        // architecture as DIM 5 (nl=1/ch=16/FLAT/lr=0.0015) with bs
        // scaled up to 32 to match DIM 5's gradient-update cadence
        // (DIM 6 has 2x training samples, so bs must double to keep
        // total updates ~constant).  NRMSE 0.003346, -36% vs Ridge raw,
        // -10% vs Ridge translated.
        p.cnn.num_layers    = 1;
        p.cnn.conv_channels = 16;
        p.cnn.readout_type  = HCNNReadoutType::FLATTEN;
        p.cnn.epochs        = 2000;
        p.cnn.batch_size    = 32;
        p.cnn.lr_max        = 0.0015f;
    }
    else if constexpr (DIM == 7)
    {
        // DIM 7 GOLD STANDARD (frozen 2026-04-14, runs 27-30):
        //   nl=2, ch=24, FLAT, ep=2000, bs=64, lr=0.0015
        //   → NRMSE 0.001494 (10 CNN seeds on the survey reservoir),
        //     **-52.8% vs Ridge translated (0.00316)**,
        //     **-65.3% vs Ridge raw (0.00431)**.  Runtime ~262s.
        //
        // Two DIM 5/6 architectural invariants broke at DIM 7:
        //   1. nl=1 is no longer optimal — nl=2 beats nl=1 by 32%.
        //   2. ch=16 is no longer the "absolute constant" — ch=24
        //      beats ch=16 by 16% at nl=2.
        // Both breaks point the same direction: DIM 7's 128-neuron
        // reservoir has the capacity budget to absorb more parameters
        // than DIM 5/6 did, so the small-N sweet spots (flat + lean)
        // stop being optimal.
        //
        // bs=64 holds the ~50k gradient-update invariant (DIM 7 has
        // ~1613 training samples → ~25 updates/epoch × 2000 epochs
        // = ~50k total updates, matching DIM 5/6 Gold cadence).
        //
        // 5→10 seed drift from scouting to confirmation was +6.9%
        // (5-seed 0.001398 → 10-seed 0.001494), higher than DIM 6's
        // 1.4-3.0% drift but still decisively the best config on
        // the DIM 7 leaderboard by a wide margin.  Pareto-frontier
        // slot (cheap-knee config) still TBD.
        //
        // See `docs/HCNNTuning.md` DIM 7 section for runs 27-30
        // narrative and the full scouting frontier.
        p.cnn.num_layers    = 2;
        p.cnn.conv_channels = 24;
        p.cnn.readout_type  = HCNNReadoutType::FLATTEN;
        p.cnn.epochs        = 2000;
        p.cnn.batch_size    = 64;
        p.cnn.lr_max        = 0.0015f;
    }
    // DIM 8+: untuned.  Falls through to CNNReadoutConfig defaults.

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

// ---------------------------------------------------------------------------
//  HRCCNN baseline — uniform first-probe architecture, DIM 5-16.
// ---------------------------------------------------------------------------

/// @brief HRCCNN baseline HCNN config — see `docs/HRCCNNBaselineConfig.md`.
///
/// Minimum-capacity first-probe architecture, uniform across every DIM:
///
///   nl=1, ch=8, FLATTEN, ep=2000, lr_max=0.0015, bs = 1 << (DIM-1)
///
/// `ep=2000` is calibrated for **chaotic** signals (Mackey-Glass, NARMA).
/// Smooth-signal tasks saturate at `ep=25`; do not reuse this config as-is
/// for smooth tasks without lowering the epoch count.
///
/// The `bs` formula holds the ~50k-gradient-update invariant constant
/// across DIMs and coincides with the DIM 5/6/7 Gold Standard `bs` values,
/// so benchmark cadence is commensurable with the tuned reference runs.
/// This is the current default HCNN config used by the diagnostic
/// benchmarks (`MackeyGlass<DIM>::BenchmarkCNNConfig()` and
/// `NARMA10<DIM>::BenchmarkCNNConfig()`).  The per-DIM Gold Standards in
/// `MackeyGlass()` above remain available for explicit users.
template <size_t DIM>
CNNReadoutConfig HRCCNNBaseline()
{
    CNNReadoutConfig cfg;
    cfg.num_layers    = 1;
    cfg.conv_channels = 8;
    cfg.readout_type  = HCNNReadoutType::FLATTEN;
    cfg.epochs        = 2000;
    cfg.batch_size    = 1 << (DIM - 1);
    cfg.lr_max        = 0.0015f;
    return cfg;
}

}  // namespace hcnn_presets
