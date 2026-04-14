// Driver for OptimizeHRCCNNForMG — edited per iteration as we tune each DIM.
//
// Current target: DIM 5 Mackey-Glass horizon 1, fine lr_max refinement
// around the frozen preset value (lr=0.003).
//
// Run 18 (coarse decade sweep, 5 seeds) traced a clean U with a crisp
// minimum at lr=0.003 — factor-of-2 to either side cost 17-45% NRMSE.
// This run tightens the grid to ±17% around 0.003 on the low side
// ({0.002, 0.0025}) and +17% on the high side ({0.0035}), bumped to
// 10 CNN seeds for a tighter variance estimate.  The anchor lr=0.003
// already has 10-seed coverage from run 16 (0.003514), so it isn't
// re-run here.  All three new trials share nl=1/ch=16/FLAT/ep=2000/bs=16
// from the frozen preset.
//
// At 10 seeds the variance band is ~0.3-0.5%, so deltas of 2-3% vs
// the anchor should start being trustworthy.

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "OptimizeHRCCNNForMG.h"
#include "../readout/HCNNPresets.h"

int main()
{
    constexpr size_t DIM = 5;

    OptimizeHRCCNNForMG<DIM> opt(/*num_seeds=*/1, /*num_cnn_seeds=*/10);
    opt.PrintHeader();

    // Both Ridge baselines: raw (apples-to-apples with HCNN) and translated
    // (2.5N hand-crafted features, upper-bound target for HCNN to beat).
    std::cout << "\n  Ridge raw        NRMSE: " << opt.RidgeBaseline()        << "\n";
    std::cout << "  Ridge translated NRMSE: " << opt.TranslationBaseline() << "\n";

    // Frozen preset now has nl=1/ch=16/FLAT/ep=2000/bs=16/lr=0.003 baked in;
    // all trials share the full preset and only lr_max varies.
    const CNNReadoutConfig preset = hcnn_presets::MackeyGlass<5>().cnn;

    auto make = [&preset](float lr_max) {
        CNNReadoutConfig c = preset;
        c.lr_max = lr_max;
        return c;
    };

    std::vector<std::pair<std::string, CNNReadoutConfig>> trials = {
        {"lr0.0015", make(0.0015f)},
    };
    opt.RunSweep(trials);

    OptimizeHRCCNNForMG<DIM>::PrintCompletion();
    return 0;
}
