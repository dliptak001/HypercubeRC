// Driver for OptimizeHRCCNNForMG — edited per iteration as we tune each DIM.
//
// Current target: DIM 6 Mackey-Glass horizon 1, batch_size sweep.
//
// Locked from prior DIM 6 probes: nl=1/ch=16/FLAT/ep=2000/lr=0.0015.
// 10-seed variance check landed at 0.003414 — beats Ridge raw by 34%
// and Ridge translated by 8%.
//
// bs was inherited uncritically from the DIM 5 Gold Standard, but at
// DIM 6 there are 2x more training samples (806 vs 403) which doubles
// the per-epoch gradient update count at the same bs.  So bs=16 at
// DIM 6 = ~50 updates/epoch vs DIM 5's ~25 updates/epoch — different
// training cadence, may not be optimal on the new sample budget.
//
// Sweep (single axis, powers of 2):
//   bs ∈ {8, 16, 32, 64}
//
// num_cnn_seeds=5 for a quick scouting pass; if the curve points
// somewhere other than bs=16, refine at 10 seeds next.

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "OptimizeHRCCNNForMG.h"
#include "../readout/HCNNPresets.h"

int main()
{
    constexpr size_t DIM = 6;

    OptimizeHRCCNNForMG<DIM> opt(/*num_seeds=*/1, /*num_cnn_seeds=*/10);
    opt.PrintHeader();

    // Both Ridge baselines: raw (apples-to-apples with HCNN) and translated
    // (2.5N hand-crafted features, upper-bound target for HCNN to beat).
    std::cout << "\n  Ridge raw        NRMSE: " << opt.RidgeBaseline()        << "\n";
    std::cout << "  Ridge translated NRMSE: " << opt.TranslationBaseline() << "\n";

    // DIM 6 Gold Standard with ep=1500 instead of the frozen ep=2000.
    // Single-point check: does dropping 25% of the epoch budget cost
    // any meaningful NRMSE on the new backbone?
    CNNReadoutConfig c;
    c.num_layers    = 1;
    c.conv_channels = 16;
    c.readout_type  = HCNNReadoutType::FLATTEN;
    c.epochs        = 1500;
    c.batch_size    = 32;
    c.lr_max        = 0.0015f;

    std::vector<std::pair<std::string, CNNReadoutConfig>> trials = {
        {"ep1500-10seed", c},
    };
    opt.RunSweep(trials);

    OptimizeHRCCNNForMG<DIM>::PrintCompletion();
    return 0;
}
