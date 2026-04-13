// Driver for OptimizeHRCCNNForMG — edited per iteration as we tune each DIM.
//
// Current target: DIM 8, Mackey-Glass horizon 1.
// DIM 5 is frozen in readout/HCNNPresets.h as nl=3/ch=32/ep=2000/bs=16.
// DIM 6 partial (2026-04-13): width is the dominant lever — nl3/ch64/ep500
// = 0.00672 beat both nl3/ch32/ep2000 and nl4/ch32/ep500 (both stuck at
// 0.0078). Not yet below Ridge raw 0.00520.

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "OptimizeHRCCNNForMG.h"
#include "../readout/HCNNPresets.h"

int main()
{
    constexpr size_t DIM = 8;

    OptimizeHRCCNNForMG<DIM> opt(/*num_seeds=*/1, /*num_cnn_seeds=*/2);
    opt.PrintHeader();

    // Both Ridge baselines: raw (apples-to-apples with HCNN) and translated
    // (2.5N hand-crafted features, upper-bound target for HCNN to beat).
    std::cout << "\n  Ridge raw        NRMSE: " << opt.RidgeBaseline()        << "\n";
    std::cout << "  Ridge translated NRMSE: " << opt.TranslationBaseline() << "\n";

    // Run 17 chunk 1: DIM 8 landscape probe at ep=200 for fast feedback.
    // DIM 8 is N=256 with ~3226 train samples — enough capacity to host
    // ch=64 and enough depth for nl up to DIM-2 = 6 (the real HCNNConv
    // ceiling; the auto-rule min(DIM-3,4)=4 is off-by-one low).
    //
    // Three orthogonal probes, each holds two axes fixed so we can
    // attribute deltas to a single lever:
    //   1. dim5-shape   nl=3 ch=32 — reference floor, matches DIM 5 winner.
    //   2. ch64-wide    nl=3 ch=64 — pure width probe (DIM 6 signal said
    //                                width is the dominant lever).
    //   3. nl5-deep     nl=5 ch=32 — pure depth probe past the auto-rule,
    //                                one short of the hard DIM-2=6 ceiling.
    //
    // Chunk 2 (tomorrow) will combine the winning levers at ep=2000.
    const CNNReadoutConfig dim5_winner = hcnn_presets::MackeyGlass<5>().cnn;

    // bs=128 matches DIM 5's ~25 updates/epoch cadence (3225/128 ≈ 25).
    // Less total gradient work than bs=16/ep=200 (5k vs 40k steps), but
    // each epoch is ~8× cheaper so wall time drops substantially.  Fine
    // for directional probing of the depth-vs-width scaling landscape.
    CNNReadoutConfig dim5_shape_ep200 = dim5_winner;
    dim5_shape_ep200.epochs     = 200;
    dim5_shape_ep200.batch_size = 128;

    CNNReadoutConfig ch64_wide_ep200 = dim5_winner;
    ch64_wide_ep200.conv_channels = 64;
    ch64_wide_ep200.epochs        = 200;
    ch64_wide_ep200.batch_size    = 128;

    CNNReadoutConfig nl5_deep_ep200 = dim5_winner;
    nl5_deep_ep200.num_layers = 5;   // auto-rule caps at 4, real ceiling 6
    nl5_deep_ep200.epochs     = 200;
    nl5_deep_ep200.batch_size = 128;

    std::vector<std::pair<std::string, CNNReadoutConfig>> trials = {
        {"dim5-shape-ep200", dim5_shape_ep200},
        {"ch64-wide-ep200",  ch64_wide_ep200},
        {"nl5-deep-ep200",   nl5_deep_ep200},
    };
    opt.RunSweep(trials);

    OptimizeHRCCNNForMG<DIM>::PrintCompletion();
    return 0;
}
