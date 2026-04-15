// Driver for OptimizeHRCCNNForMG — edited per iteration as we tune each DIM.
//
// Current run: DIM 5..7 baseline pass, nl=1/ch=8/FLAT, lr=0.0015, ep=2000,
// num_cnn_seeds=1.  Completes the baseline-config picture below the Gold
// Standards already frozen at DIM 5/6/7 — shows what the lean first-probe
// architecture delivers at the small end, where nl=1/ch=16 (DIM 5/6) or
// nl=2/ch=24 (DIM 7) is the tuned optimum.
//
// Batch size: bs = 1 << (DIM - 1), doubling per DIM to preserve the ~50k
// gradient-update invariant (train_samples ~= 12.6 * N, so bs ∝ N keeps
// updates/epoch ~= 25 constant).  This matches the DIM 5/6/7 Gold cadence
// (DIM 5 Gold bs=16, DIM 6 Gold bs=32, DIM 7 Gold bs=64) and scales
// uniformly across DIM 5-14.
//
// Rationale: reservoir collection is cheap compared to RCNN training, so
// the training-cost knob is bs.  Scaling bs with N makes the comparison
// apples-to-apples across DIM.
//
// bs table:
//   DIM  5: bs   16  (train  403,   25 upd/ep)
//   DIM  6: bs   32  (train  806,   25 upd/ep)
//   DIM  7: bs   64  (train 1613,   25 upd/ep)
//   DIM  8: bs  128  (train 3225,   25 upd/ep)
//   DIM  9: bs  256  (train 6451,   25 upd/ep)
//   DIM 10: bs  512  (train 12902,  25 upd/ep)
//   DIM 11: bs 1024  (train 25804,  25 upd/ep)
//   DIM 12: bs 2048  (train 51609,  25 upd/ep)
//   DIM 13: bs 4096  (train 103219, 25 upd/ep)
//   DIM 14: bs 8192  (train 206438, 25 upd/ep)
//
// Seed note: hcnn_presets::MackeyGlass<DIM>() has MG-surveyed seeds for
// DIM 5-8, so DIM 5..7 first-probe numbers will use the same reservoirs
// as the frozen Gold Standards and are directly commensurable with them.

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "OptimizeHRCCNNForMG.h"
#include "../readout/HCNNPresets.h"

template <size_t DIM>
static void RunDim()
{
    OptimizeHRCCNNForMG<DIM> opt(/*num_seeds=*/1, /*num_cnn_seeds=*/1);
    opt.PrintHeader();

    CNNReadoutConfig cfg;
    cfg.num_layers    = 1;
    cfg.conv_channels = 8;
    cfg.readout_type  = HCNNReadoutType::FLATTEN;
    cfg.epochs        = 2000;
    cfg.batch_size    = 1 << (DIM - 1);
    cfg.lr_max        = 0.0015f;

    const std::string label = "dim" + std::to_string(DIM) + "-nl1-ch8";
    opt.RunSweep({{label, cfg}});
    OptimizeHRCCNNForMG<DIM>::PrintCompletion();
    std::cout << std::endl;
}

template <size_t... Is>
static void RunAll(std::index_sequence<Is...>)
{
    (RunDim<Is + 5>(), ...);  // DIM 5..7
}

int main()
{
    RunAll(std::make_index_sequence<3>{});
    return 0;
}
