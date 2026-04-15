// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

/// @file HCNNProfileDriver.cpp
/// @brief Standalone HCNN training profiler.
///
/// Runs a single HCNN training pass under the HRCCNN baseline config
/// on Mackey-Glass h=1, then prints the cumulative per-phase breakdown
/// from `hcnn::profile`.  Independent of `OptimizeHRCCNNForMG` — this
/// driver exists purely to answer "where is HCNN training time going?"
/// and has no tuning responsibilities.
///
/// Profiling infrastructure lives in `HypercubeCNN/HCNNProfile.h`.  To
/// make this driver actually collect numbers, set `HCNN_PROFILE` to 1
/// in that header and rebuild `HypercubeCNNCore`.  With `HCNN_PROFILE`
/// at its default of 0, the Reset/Report calls below are no-ops and
/// the driver just runs the training pass without any timing overhead.
///
/// Default target: DIM 8 baseline (`nl=1, ch=8, FLAT, ep=2000, bs=128,
/// lr=0.0015`) — picked because it's the smallest shape where the
/// O(N²)-per-epoch regime kicks in (see `docs/HRCCNNBaselineConfig.md`
/// observation 3), so the profile reflects the actual high-DIM hot
/// path we want to optimize, and it completes in a manageable ~50s.
/// Override the DIM via the `HCNN_PROFILE_DIM` macro at compile time
/// if a different shape is desired.

#include <cstddef>
#include <iostream>

#include "OptimizeHRCCNNForMG.h"
#include "../readout/HCNNPresets.h"
#include "HCNNProfile.h"

#ifndef HCNN_PROFILE_DIM
#define HCNN_PROFILE_DIM 8
#endif

int main()
{
    constexpr size_t DIM = HCNN_PROFILE_DIM;

    std::cout << "=== HCNNProfileDriver ===\n"
              << "  target: DIM " << DIM
              << "  HRCCNN baseline config  MG h=1\n";
#if HCNN_PROFILE
    std::cout << "  HCNN_PROFILE is ENABLED — per-phase timing will be collected.\n\n";
#else
    std::cout << "  HCNN_PROFILE is DISABLED — set HCNN_PROFILE=1 in\n"
              << "  HypercubeCNN/HCNNProfile.h and rebuild HypercubeCNNCore\n"
              << "  to collect per-phase timing.\n\n";
#endif

    OptimizeHRCCNNForMG<DIM> opt(/*num_seeds=*/1, /*num_cnn_seeds=*/1);
    opt.PrintHeader();

    CNNReadoutConfig cfg = hcnn_presets::HRCCNNBaseline<DIM>();

    // Call RunOne directly (bypassing RunSweep) to skip the Ridge baseline
    // computation.  Profiling is about HCNN training time — Ridge is
    // irrelevant and adds noise to the wall-clock budget.
    std::cout << "\n"
              << "  label                 | layers | ch | head |  ep  |  bs | lr_max  |    NRMSE |  time(s)\n"
              << "  ----------------------+--------+----+------+------+-----+---------+----------+---------\n";
    hcnn::profile::Reset();
    opt.RunOne(cfg, "profile-dim" + std::to_string(DIM));
    OptimizeHRCCNNForMG<DIM>::PrintCompletion();

    hcnn::profile::Report(std::cout);
    return 0;
}
