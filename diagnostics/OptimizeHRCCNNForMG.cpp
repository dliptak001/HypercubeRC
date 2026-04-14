// Driver for OptimizeHRCCNNForMG — edited per iteration as we tune each DIM.
//
// Current target: DIM 8 Mackey-Glass horizon 1, first probe (run 31).
//
// Minimum-viable DIM 8 first probe: pipeline check + directional read
// on the nl=2 transfer question.
//
// Trial: nl2-ch16/FLAT/ep=2000/bs=128/lr=0.0015, 5 CNN seeds.
//
// Calibration:
//   - bs=128 with 3226 training samples → 25 updates/ep × 2000 ep
//     = 50k total updates.  Matches the gradient-update invariant used
//     at DIM 5/6/7 Gold — this is properly calibrated, not under-trained.
//   - 5 seeds is the standard scouting resolution.  Good for ranking
//     configs with ≥5-10% deltas; 10-seed refinement only for freezes.
//
// Expected runtime: ~8.5 min (DIM 7 nl=2/ch=16 @ ep=2000/5seeds was
// 128s; scale ~4× for DIM 8 → ~512s).

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "OptimizeHRCCNNForMG.h"
#include "../readout/HCNNPresets.h"

int main()
{
    constexpr size_t DIM = 8;

    OptimizeHRCCNNForMG<DIM> opt(/*num_seeds=*/1, /*num_cnn_seeds=*/5);
    opt.PrintHeader();

    std::cout << "\n  Ridge raw        NRMSE: " << opt.RidgeBaseline()        << "\n";
    std::cout << "  Ridge translated NRMSE: " << opt.TranslationBaseline() << "\n";

    CNNReadoutConfig nl2_ch16;
    nl2_ch16.num_layers    = 2;
    nl2_ch16.conv_channels = 16;
    nl2_ch16.readout_type  = HCNNReadoutType::FLATTEN;
    nl2_ch16.epochs        = 2000;
    nl2_ch16.batch_size    = 128;
    nl2_ch16.lr_max        = 0.0015f;

    std::vector<std::pair<std::string, CNNReadoutConfig>> trials = {
        {"nl2-ch16-first-probe", nl2_ch16},
    };
    opt.RunSweep(trials);

    OptimizeHRCCNNForMG<DIM>::PrintCompletion();
    return 0;
}
