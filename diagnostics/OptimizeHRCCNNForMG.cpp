// Driver for OptimizeHRCCNNForMG — edited per iteration as we tune each DIM.
//
// Current target: DIM 6, Mackey-Glass horizon 1.
// DIM 5 is frozen in readout/HCNNPresets.h as nl=3/ch=32/ep=2000/bs=16.

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "OptimizeHRCCNNForMG.h"
#include "../readout/HCNNPresets.h"

int main()
{
    constexpr size_t DIM = 6;

    OptimizeHRCCNNForMG<DIM> opt(/*num_seeds=*/1, /*num_cnn_seeds=*/3);
    opt.PrintHeader();

    // Both Ridge baselines: raw (apples-to-apples with HCNN) and translated
    // (2.5N hand-crafted features, upper-bound target for HCNN to beat).
    std::cout << "\n  Ridge raw        NRMSE: " << opt.RidgeBaseline()        << "\n";
    std::cout << "  Ridge translated NRMSE: " << opt.TranslationBaseline() << "\n";

    // Run 16: DIM 6 chunk 1 — test the DIM 5 winner's hyperparameters at
    // DIM 6, plus the two scaling-direction probes (more depth, more width).
    // The DIM 5 winner was nl=3/ch=32/ep=2000/bs=16/lr=0.003.  At DIM 6:
    //   - nl ≤ 4 is the real depth ceiling (DIM - 2), not the auto nl=3.
    //   - train samples = ~806 (vs 403 at DIM 5), so ch=64 may now be
    //     viable without overfitting.
    //
    // Starting from the DIM 5 preset so the exact hyperparameters match,
    // then varying num_layers and conv_channels in the follow-up trials.
    const CNNReadoutConfig dim5_winner = hcnn_presets::MackeyGlass<5>().cnn;

    CNNReadoutConfig dim5_applied = dim5_winner;  // nl=3/ch=32/ep=2000/bs=16

    CNNReadoutConfig nl4_ch32_ep2000 = dim5_winner;
    nl4_ch32_ep2000.num_layers = 4;   // depth probe (DIM 6 allows up to nl=4)

    CNNReadoutConfig nl3_ch64_ep2000 = dim5_winner;
    nl3_ch64_ep2000.conv_channels = 64;  // width probe (more samples at DIM 6 may tolerate it)

    std::vector<std::pair<std::string, CNNReadoutConfig>> trials = {
        {"dim5-winner",    dim5_applied},
        {"nl4-ch32-ep2000", nl4_ch32_ep2000},
        {"nl3-ch64-ep2000", nl3_ch64_ep2000},
    };
    opt.RunSweep(trials);

    OptimizeHRCCNNForMG<DIM>::PrintCompletion();
    return 0;
}
