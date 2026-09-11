#pragma once

#include <cstddef>
#include <cstdint>

#include "Reservoir.h"
#include "Readout.h"

/// @file Presets.h
/// @brief Per-DIM preset bundles: surveyed reservoir seed + baseline readout config.
///
/// `Baseline<DIM>()` returns both the surveyed reservoir seed and
/// the baseline CNN architecture for that DIM. Callers use `.reservoir`
/// and `.cnn` to extract the pieces they need.

namespace presets {

/// Reservoir config + readout config, bundled per-DIM.
struct Preset
{
    ReservoirConfig   reservoir;
    ReadoutConfig cnn;
};

/// @brief Baseline preset: surveyed reservoir seed + uniform CNN architecture.
///
/// CNN architecture: nl=1, ch=8, ep=2000, lr_max=0.0015, bs=1<<(DIM-1).
/// Reservoir seed: per-DIM 500-seed survey winner via `SurveyedSeed<DIM>()`.
/// CNN seed: per-DIM weight-init survey winner (DIM 5-10; default 42 above).
///
/// `ep=2000` targets chaotic signals (NARMA). Override `epochs` for
/// non-chaotic tasks.
template <size_t DIM>
Preset Baseline()
{
    Preset p;
    p.reservoir.seed = ::SurveyedSeed<DIM>();

    p.cnn.num_layers    = 1;
    p.cnn.conv_channels = 8;
    p.cnn.epochs        = 2000;
    p.cnn.batch_size    = 1 << (DIM - 1);
    p.cnn.lr_max        = 0.0015f;

    if constexpr      (DIM == 5)  p.cnn.seed = 21;
    else if constexpr (DIM == 6)  p.cnn.seed = 34;
    else if constexpr (DIM == 7)  p.cnn.seed = 6;
    else if constexpr (DIM == 8)  p.cnn.seed = 2;
    else if constexpr (DIM == 9)  p.cnn.seed = 20;
    else if constexpr (DIM == 10) p.cnn.seed = 2;
    return p;
}

}  // namespace presets
