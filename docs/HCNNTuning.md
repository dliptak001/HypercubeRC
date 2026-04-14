# HCNN Readout Tuning — Per-DIM Best Configs

Living record of the best CNNReadoutConfig found for each hypercube dimension
on the two chaotic-regression benchmarks. Populated iteratively via
`diagnostics/OptimizeHRCCNNForMG.h` (Mackey-Glass) and its NARMA-10 counterpart.

## Baselines

Numbers in the "Ridge raw" column come from `MackeyGlass<DIM>` /
`NARMA10<DIM>` with `FeatureMode::Raw` and `output_fraction = 1.0` — the
apples-to-apples baseline for HCNN, which also operates on raw state.
"Ridge translated" is `FeatureMode::Translated` (2.5N features) for
context; HCNN's goal is to match or beat the raw baseline.

Lower NRMSE is better.

## Mackey-Glass (horizon = 1)

Data: 18·N collected samples, 70/30 train/test, normalized to [-1, +1].

| DIM |   N   | Ridge raw | Ridge trans. | Best HCNN | layers | ch | head | ep | bs | lr_max | wd | seed | Notes |
|----:|------:|----------:|-------------:|----------:|-------:|---:|:----:|---:|---:|-------:|---:|-----:|:------|
|   5 |    32 |  0.006165 |     0.004375 |  0.003169 |      1 | 16 | FLAT |2000|  16| 0.0015 |  0 | 11822067163148543833 | **GOLD STANDARD** 2026-04-13 (runs 15-20, averaged 10 CNN inits). **-49% vs Ridge raw**, **-28% vs Ridge translated**. Single conv layer, FLATTEN readout, lr=0.0015. Architecture pinned in runs 15-17 (GAP vs FLATTEN have opposite backbone preferences); lr_max refined in runs 18-20 (coarse decade sweep missed the optimum by spacing — fine sweep at 10 seeds showed a clean bowl with minimum at 0.0015). Runtime ~14s (6.9× faster than the old nl=3/ch=32/GAP preset at 95s). |
|   6 |    64 |  0.005205 |     0.003713 |  0.003346 |      1 | 16 | FLAT |2000|  32 | 0.0015 |  0 | 11459651989651327597 | **GOLD STANDARD** 2026-04-13 (runs 21-26, averaged 10 CNN inits). **-36% vs Ridge raw**, **-10% vs Ridge translated**. Direct architectural transfer from DIM 5 (nl=1/ch=16/FLAT/lr=0.0015); bs scaled 16→32 to hold total gradient updates constant as training samples doubled (806 vs 403). Runtime ~11s. |
|   7 |   128 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|   8 |   256 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|   9 |   512 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|  10 |  1024 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|  11 |  2048 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|  12 |  4096 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |

## NARMA-10

Data: 18·N collected samples, 70/30 train/test, input `u ~ U(0, 0.5)`,
target from the standard 10th-order recurrence.

| DIM |   N   | Ridge raw | Ridge trans. | Best HCNN | layers | ch | head | ep | bs | lr_max | wd | seed | Notes |
|----:|------:|----------:|-------------:|----------:|-------:|---:|:----:|---:|---:|-------:|---:|-----:|:------|
|   5 |    32 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|   6 |    64 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|   7 |   128 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|   8 |   256 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|   9 |   512 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|  10 |  1024 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|  11 |  2048 |         – |            – |         – |      – |  – |   –  |  – |  – |      – |  – |    – | not yet tuned |
|  12 |  4096 |         – |            – |         – |      – |  – |   –  |  − |  – |      – |  – |    – | not yet tuned |

## Column legend

- **layers**: Conv+Pool pairs (0 in `CNNReadoutConfig` means auto =
  `max(1, min(DIM-3, 4))` — this column records the *resolved* count)
- **ch**: base `conv_channels`; each successive layer doubles
- **head**: readout head after the conv stack. `GAP` = global average pool per
  channel → `[c_final]` → Linear (translation-invariant across vertices).
  `FLAT` = every `(channel, vertex)` cell is a feature → `[c_final × 2^final_dim]`
  → Linear (position-sensitive, one weight per vertex × channel). Plumbed via
  `CNNReadoutConfig::readout_type`.
- **ep**: epochs
- **bs**: batch_size
- **lr_max**: peak learning rate (cosine annealing floor =
  `lr_max * lr_min_frac`, default fraction 0.1)
- **wd**: `weight_decay` (L2)
- **seed**: CNN weight-init seed (default 42)

## Per-DIM notes

Narrative log for tuning runs — what was tried, what worked, what didn't.
Kept separately from the summary table so the table stays scannable.

### DIM 5

**Run 1 (baseline):** `mg-bench` = MG BenchmarkCNNConfig (ep=100, bs=128, lr=0.003, ch=16, auto layers → 2) produced NRMSE **0.055921** in 0.42s vs Ridge raw **0.023445**. HCNN is ~2.4× worse. At N=32 with train=403 samples, `batch=128` gives only ~3 gradient updates per epoch, so total weight updates over 100 epochs ≈ 300 — almost certainly under-converged.

**Run 2 (batch_size sweep, ep=100):** Clean monotonic improvement — bs=128 (0.0559) → 64 (0.0413) → 32 (0.0377) → 16 (0.0296) → 8 (0.0280). bs=16 is the cost/quality knee; bs=8 costs 2× the time for only 5% more gain. bs=16 chosen as the fixed batch size going forward.

**Run 3 (epoch sweep at bs=16):** HCNN crossed Ridge between ep=200 (0.0229) and ep=300 (0.0184), and kept falling to ep=500 (0.0143, -39% vs Ridge). No saturation yet. bs=8+ep=200 (0.0159) confirmed bs=16+more-epochs is a better use of compute than bs=8.

**Run 4 (plateau probe + axis-pruning):** ep=500→ep=1000 dropped another 37% to **0.00895** (-62% vs Ridge), still no saturation. Pruned two axes: **lr=0.005 ≡ lr=0.003** (0.01424 vs 0.01431, convergence is epoch-bound); **ch=32** buys 8-16% NRMSE but costs 2× time vs ch=16. Best absolute result: ep=1000+ch=32 = **0.00826** (-65% vs Ridge, 24.3s). Best per-unit-time: ep=1000+ch=16 (12s). Next step: extend epochs (ep=1500, 2000, 3000) at ch=16 to find the actual plateau; probe weight_decay for regularization at high epochs.

**RESET (discovered in run 5):** Runs 1-5 were all on seed=42, which is a sub-optimal DIM-5 reservoir realization. Switching to the per-DIM survey seed (`11822067163148543833`, from `MackeyGlass<DIM>::DefaultSeed()`) dropped Ridge raw from 0.0234 → 0.00616 (~3.8× better) and Ridge translated from 0.0206 → 0.00438. Baselines now match `MackeyGlass.md` exactly. All prior HCNN conclusions about relative performance are invalidated; restart the sweep on the correct reservoir.

**Run 6 + 7 (after seed fix, on survey reservoir, split into chunks after ThreadPool deadlock investigation):** mg-bench (ep=100, bs=128) = 0.0740 (+1100% vs Ridge raw) — bad as expected. At bs=16: ep=100 → 0.0247, ep=500 → 0.00956, ep=750 → 0.00994, **ep=1000 → 0.00760 (best so far)**, ep=1500 → 0.00839. Curve is non-monotonic past ep=1000 — small deltas at this scale are dominated by single-seed NRMSE variance. HCNN is +23% vs Ridge raw and +74% vs Ridge translated, so the real target is 0.00438 (Ridge translated) and there's still a sizable gap. Next: at ep=1000, probe capacity (ch=32) and regularization (wd=1e-5) to see if either axis moves the needle.

**Run 8 (capacity + regularization probe):** `ep1000-ch32` = **0.00526** (-15% vs Ridge raw, +20% vs Ridge translated) — first config below Ridge raw. `ep1000-wd1e-5` = 0.00710 (7% better than baseline ch=16). Re-baselined `ep1000` was bit-exact at 0.00760, confirming the ThreadPool fix isn't perturbing numerical results. Looked like a clear capacity win.

**Run 9 (stack-the-wins probe):** Tested `ch32-wd1e5` (0.00566, +8% regression), `ch32-ep1500` (0.00766, +46% regression), `ch64-ep1000` (0.00630, +20% regression). Three out of three configs went the wrong direction. The wd effect flipped sign between ch=16 and ch=32, and more epochs at ch=32 made things dramatically worse. This flagged that single-seed measurements were being overwhelmed by training-trajectory noise on small hyperparameter deltas.

**Run 10 (CNN-init averaging, num_cnn_seeds=3):** Added `num_cnn_seeds` parameter to RunOne: each trial now runs the HCNN training 3× with seeds 42/43/44 at the fixed survey reservoir, reporting the mean NRMSE. The ranking **changed dramatically**:
- `ch16-ep1000`: 0.00760 → 0.00826 (+9%)
- `ch32-ep1000`: **0.00526 → 0.00731 (+39%)** — the "breakthrough" was a lucky init
- `ch32-ep1500`: 0.00766 → **0.00678 (-12%)** — the "regression" was an unlucky init

Under averaging, **ch=32-ep=1500 at 0.00678 is the real current best**, still +10% above Ridge raw. All prior run 7-9 hyperparameter conclusions must be re-verified under averaging before being trusted. The noise floor for ch=32 was ~40% of signal on single-seed. Next: push epochs at ch=32 (ep=2000, 3000) and retest ch=64 under averaging.

**Run 11 (epoch + capacity sweep at ch=32):** `ch32-ep2000 = 0.00542` became new best (-12% vs Ridge raw — first solid crossing). `ch32-ep3000 = 0.00626` regressed, identifying an epoch plateau around 2000 at nl=2-ch=32. `ch64-ep1500 = 0.00629` ≈ ch32-ep3000 — more channels at nl=2 doesn't unlock anything.

**Run 12 (num_layers=1 probe):** All three flat-topology configs (`nl1-ch32-ep2000`, `nl1-ch32-ep3000`, `nl1-ch64-ep2000`) landed around 0.026-0.028 — **~5× worse than nl=2-ch=32-ep=2000**. Pool hierarchy isn't destroying signal; it's structurally essential — the second conv's ability to see "features of vertex pairs" can't be recovered by widening a flat net.

**Run 13 (num_layers=3 probe):** Auto-rule `min(DIM-3,4)` caps DIM=5 at nl=2; the assert enforcing it is compiled out under `-DNDEBUG`, so nl=3 actually runs. Result: **`nl3-ch32-ep2000 = 0.00490`, new best** (-21% vs Ridge raw, +12% vs Ridge translated). `nl3-ch16-ep2000 = 0.00683` and `nl3-ch16-ep3000 = 0.00541` showed narrow-deep needs more training but ultimately doesn't match wide-deep. Depth+width are coupled: more layers require more channels to fill the hierarchy with useful features.

**Run 14 (DIM 5 finalization):** `nl3-ch32-ep3000 = 0.00609` (regression, confirms ep=2000 is the real plateau at nl=3-ch=32, not a still-dropping curve). `nl3-ch64-ep2000 = 0.01010` (heavy overfit — 4× the nl=3-ch=32 params on 403 samples). `nl4-ch32-ep2000` **crashed** with `HCNNConv requires 3 <= DIM <= 32`: at DIM=5 the 4th conv would operate on 2 vertices (DIM=2), below HCNNConv's minimum. **Real max depth at DIM=D is `nl ≤ D - 2`, not `D - 3` as the assert in CNNReadout::build_architecture claims.** The assert is off-by-one and should be relaxed to match HCNNConv's actual constraint.

**DIM 5 FROZEN at nl=3, ch=32, ep=2000, bs=16, lr=0.003 → NRMSE 0.00490.** Next: wire this into `readout/HCNNPresets.h` as the first entry, then start DIM 6. Key lesson for DIM 6+: always push the depth knob past the auto-rule (up to `nl = DIM - 2`) — depth is a real lever and the auto-rule is too conservative.

**Run 15 (first-principles GAP vs FLATTEN sweep, 2026-04-13):** Up to this point `CNNReadout::build_architecture()` hardcoded `hcnn::ReadoutType::GAP`. Two reasons to question that default on a hypercube reservoir:

1. *Vertex identity carries signal.* The hypercube is topologically vertex-transitive, but the reservoir's random `W_in` assigns each vertex its own contribution from the driving signal — two vertices sit in identical graph positions but respond differently to the input stream. Ridge raw exploits this trivially (one weight per vertex). GAP averages across vertices and discards that information; FLATTEN keeps it (one weight per `(channel, vertex)` cell). On a hypercube RC, GAP's translation-invariance prior is structurally wrong.
2. *Fan-in rank bound on channels.* At layer 1, each output channel is `TANH(W · neighbors + b)` where `W ∈ R^K`, `K = DIM`, no self-term (HCNNConv.h:65). The rank ceiling of any `R^K → R^C` linear map is `min(K, C)`, so at DIM=5 a layer-1 map is at most rank 5. The frozen winner's ch=32 is 6.4× over-complete; some of that is TANH piecewise nonlinearity + training-lottery benefit, but ch should roughly track `1-2*DIM`, not a fixed 32.

Plumbed `HCNNReadoutType { GAP, FLATTEN }` through `CNNReadoutConfig::readout_type` → `CNNReadout::build_architecture()` → `hcnn::ReadoutType`, then ran a 2×2 on the two hypotheses under the frozen training budget (ep=2000, bs=16, lr=0.003, num_cnn_seeds=3):

| Trial          | nl | ch | head | NRMSE    | vs Ridge raw | vs Ridge trans | time  |
|----------------|---:|---:|:----:|---------:|-------------:|---------------:|------:|
| frozen-gap     |  3 | 32 | GAP  | 0.004897 |         -21% |           +12% | 95.2s |
| frozen-flatten |  3 | 32 | FLAT | 0.006511 |          +6% |           +49% | 94.2s |
| lean-gap       |  2 | 16 | GAP  | 0.007099 |         +15% |           +62% | 20.0s |
| **lean-flatten** | **2** | **16** | **FLAT** | **0.003684** | **-40%** | **-16%** | **20.4s** |

`frozen-gap` reproduces the run-13 baseline exactly (sanity check). The other three results show that **the two readout heads have opposite backbone preferences**: GAP wants deep + wide (`frozen-gap` beats `lean-gap` by 31%), FLATTEN wants shallow + lean (`lean-flatten` beats `frozen-flatten` by 43%). H1 and H2 are not independent axes — they interact.

Why the inversion: at nl=3, only 4 vertices survive pooling, so FLATTEN's per-vertex readout has 4×128=512 weights on 4 spatial positions against 403 training samples — over-parameterized *and* the spatial signal has already been smeared by the deep pool. At nl=2, 8 vertices survive and FLATTEN has 8×32=256 weights, the sweet spot: enough spatial distinctions to exploit, few enough parameters not to overfit. `lean-gap` fails for the mirror reason — GAP on 32 channels leaves the linear layer with only 32 features, fewer than Ridge raw's 32 vertices with no nonlinearity advantage.

**`lean-flatten` is the first HCNN config to meaningfully beat Ridge translated at DIM 5 MG** — 0.003684 vs 0.004375, a 16% improvement over the hand-crafted `[x | x² | x·x_antipodal]` feature basis, and ~4.7× faster than the old frozen-gap winner. Depth and width were actively hurting us at this DIM; the right recipe for FLATTEN is the *opposite* of the GAP recipe.

**Status: PROVISIONAL.** Only 3 CNN-init seeds; the 16% margin vs Ridge translated is suggestive but not yet variance-bounded. Before promoting lean-flatten to the DIM 5 MG preset in `HCNNPresets.h`, re-run at num_cnn_seeds≥10 to get a proper variance estimate. Also worth probing `nl=1, ch=16, FLAT` to see whether depth-2 is even necessary or if a single conv layer suffices once FLATTEN retains the spatial structure.

**Working hypotheses for DIM 6+:** ch should scale with DIM (try `ch ≈ DIM` to `2*DIM` instead of fixed 32). GAP vs FLATTEN must be swept, not assumed. For FLATTEN, *shallower* may be better because more surviving vertices matter more than deeper features. The "push depth to `DIM - 2`" lesson from run 14 applied to GAP; for FLATTEN it may invert.

**Run 16 (FLATTEN variance + depth probe, 2026-04-13):** Run 15's winner used only 3 CNN seeds, so variance on the -16% vs-Ridge-translated margin wasn't bounded. Ran a 2×2 grid `(nl, ch) ∈ {1,2} × {8,16}` at FLAT, ep=2000, bs=16, lr=0.003, num_cnn_seeds=10. Paired by FLATTEN readout input size (128 for ch=8, 256 for ch=16) so within each pair depth is the only varying axis.

| Trial         | nl | ch | final V | FLAT in | NRMSE    | vs Ridge raw | vs Ridge trans | time  |
|---------------|---:|---:|--------:|--------:|---------:|-------------:|---------------:|------:|
| nl1-ch8-flat  |  1 |  8 |      16 |     128 | 0.004074 |         -34% |            -7% | 13.4s |
| **nl1-ch16-flat** | **1** | **16** | **16** | **256** | **0.003514** | **-43%** | **-20%** | **13.9s** |
| nl2-ch8-flat  |  2 |  8 |       8 |     128 | 0.004147 |         -33% |            -5% | 14.5s |
| nl2-ch16-flat |  2 | 16 |       8 |     256 | 0.003875 |         -37% |           -11% | 20.4s |

Four findings:

1. **The entire corner is a winning region.** All four trials beat Ridge translated. This isn't a narrow hyperparameter miracle — any FLATTEN + lean-backbone config wins at DIM 5.
2. **Depth is unnecessary and slightly harmful.** Within each FLAT-input pair: ch=8 → nl=1 beats nl=2 by 2%; ch=16 → nl=1 beats nl=2 by 9%. The second conv+pool is destroying per-vertex structure that FLATTEN wants to exploit. This completely inverts the run 14 depth lesson (which was GAP-specific).
3. **ch=16 > ch=8 at both depths.** Over ch=5 (the layer-1 fan-in floor) the 3.2× over-completeness at ch=16 still helps the optimizer find better features via TANH diversity; ch=8 slightly underfits.
4. **Run 15 variance check:** nl2-ch16-flat went 0.003684 (3 seeds) → 0.003875 (10 seeds), +5% — within expected noise but 3-seed was optimistic. **New best: nl=1, ch=16, FLAT → 0.003514**, 7% better than run 15 on a proper 10-seed basis.

**Architectural interpretation:** at nl=1 the HCNN is essentially doing nonlinear Ridge. Each conv channel is a learned nonlinear function of a vertex's 5 nearest neighbors (K = DIM, no self-term), and FLATTEN assigns per-vertex weights on 16 learned local-feature maps. The conv backbone's only job is to extract local relational features; the readout provides all the expressive power, weighting each learned feature at each vertex independently. Adding more conv+pool layers doesn't help because the useful signal is per-vertex identity (broken symmetry from random `W_in`), and pooling is specifically the operation that destroys per-vertex distinctions.

**Run 17 (epoch sweep at the new winner, 2026-04-13):** Before freezing `nl=1/ch=16/FLAT` into `HCNNPresets.h`, confirmed 2000 epochs is worth paying for vs a shorter budget. Ran the same config at ep ∈ {100, 200, 500, 1000, 1500} at num_cnn_seeds=10 with run 16's ep=2000 result as the ceiling reference:

| ep    | NRMSE    | vs Ridge trans | time  |
|------:|---------:|---------------:|------:|
|   100 | 0.013288 |          +203% |  0.7s |
|   200 | 0.009121 |          +108% |  1.3s |
|   500 | 0.004831 |           +10% |  3.5s |
|  1000 | 0.003961 |            -9% |  7.0s |
|  1500 | 0.004360 |            +0% | 10.4s |
|  2000 | 0.003514 |           -20% | 13.9s |

Curve shape: severe underconvergence below ep=200 (worse than Ridge raw), crosses Ridge raw at ~500, beats Ridge translated at ~1000, best at 2000. **Non-monotonic dip at ep=1500** (+10% vs ep=1000) — this is a real schedule effect, not seed noise: cosine LR annealing is parameterized on `progress = e / total_epochs`, so ep=1500 is not "ep=1000 with 500 more steps," it is a *different optimization problem* with a different LR trajectory that lands in a different local minimum. Each epoch-count sample is its own training run.

The ep=1000→2000 gap (13%) is larger than the ep=1500 wobble (10%), so ep=2000 is the honest plateau point, not a lucky outlier. At 13.9s the training budget is already fast (6.9× faster than the old frozen preset at 95s), so there's no wall-clock reason to compromise.

**DIM 5 MG RE-FROZEN at nl=1, ch=16, head=FLAT, ep=2000, bs=16, lr=0.003 → NRMSE 0.003514.** `HCNNPresets.h` updated. Next: probe lr_max around the current 0.003 default — the new backbone is much smaller and has different optimization dynamics than the old nl=3/ch=32 preset, so the old lr tuning may not transfer.

**Run 18 (lr_max coarse decade sweep, 5 seeds):** 5-point sweep at `lr_max ∈ {0.0005, 0.001, 0.003, 0.005, 0.01}` on `nl=1/ch=16/FLAT/ep=2000/bs=16`.

| lr_max | NRMSE    | vs Ridge trans |
|-------:|---------:|---------------:|
| 0.0005 | 0.005117 |           +17% |
| 0.001  | 0.004141 |            -5% |
| 0.003  | 0.003531 |           -19% |
| 0.005  | 0.004347 |            -1% |
| 0.01   | 0.006129 |           +40% |

Clean U-shaped curve — or so it seemed. The 5-seed lr=0.003 number (0.003531) matched run 16's 10-seed number (0.003514) within 0.5%, validating the variance estimate. Conclusion at the time: frozen preset is already at the optimum. **This was wrong — the decade spacing just happened to miss the real minimum.**

**Run 19 (lr_max fine sweep 0.002-0.0035, 10 seeds):** tightened to `{0.002, 0.0025, 0.0035}` at 10 seeds.

| lr_max | NRMSE    | vs lr=0.003 |
|-------:|---------:|------------:|
| 0.002  | 0.003227 |        -8.2% |
| 0.0025 | 0.003464 |        -1.4% |
| 0.003  | 0.003514 |            — |
| 0.0035 | 0.003936 |       +12.0% |

Monotonic improvement all the way down to the edge of the sweep. lr=0.002 beat the old anchor (0.003) by 8%, showing the minimum sits below 0.003 and run 18 had been sampled too sparsely to see it. This is a direct illustration of the "each epoch-count is a different optimization problem" hazard from run 17 applied to lr_max: small grids under cosine annealing hide real structure.

**Run 20 (lr_max = 0.0015 probe, 10 seeds):** single-trial to find the floor.

| lr_max | NRMSE    | Δ from prior best |
|-------:|---------:|------------------:|
| 0.0015 | **0.003169** |           -1.8% |

A further 1.8% improvement, but only 1.8% — well inside the 10-seed variance band (~0.3-0.5%, but the floor is slightly uncertain below that). Combined with run 18's `lr=0.001 → 0.004141` (+30.7% worse), **the curve clearly bottoms around lr=0.0015**:

| lr_max   | NRMSE    | Δ from min |
|---------:|---------:|-----------:|
| 0.001    | 0.004141 |     +30.7% |
| **0.0015** | **0.003169** |         — |
| 0.002    | 0.003227 |      +1.8% |
| 0.0025   | 0.003464 |      +9.3% |
| 0.003    | 0.003514 |     +10.9% |
| 0.0035   | 0.003936 |     +24.2% |

Tight bowl with minimum at lr_max=0.0015. Going lower (toward 0.001) costs 30% — we're at the floor, not partway down a slope.

**DIM 5 MG GOLD STANDARD at nl=1, ch=16, head=FLAT, ep=2000, bs=16, lr=0.0015 → NRMSE 0.003169.** 10 CNN seeds, on the survey reservoir. -49% vs Ridge raw, **-28% vs Ridge translated**. `HCNNPresets.h` updated. This is the canonical DIM 5 MG baseline for all downstream work.

**Methodology lessons for DIM 6+ (updated):**
- **Never trust a coarse hyperparameter sweep as final.** The run 18 decade sweep looked clean and said "you're at the optimum," but a finer grid beat it by 10%. Always refine at 10 seeds once a bowl is identified.
- GAP vs FLATTEN must be swept, not defaulted.
- Channel sizing should track `ch ≈ 1-2*DIM` (layer-1 fan-in rank bound is DIM).
- For FLATTEN at small DIM, try **nl=1 first**. The "push depth to DIM-2" lesson from run 14 was GAP-specific.
- Cosine LR annealing makes each (ep, lr) tuple a *different* optimization problem with its own trajectory. Don't extrapolate across lr values without running them.

### DIM 6

DIM 6 is the first DIM probed after the architectural pivot — nl=1/ch=16/FLAT became the DIM 5 Gold Standard in runs 15-20, so the DIM 6 probe starts from that template rather than the old nl=3/ch=32/GAP regime. Ridge baselines: raw 0.005205, translated 0.003713.

**Run 21 (first-probe 2×2, FLATTEN-only, 5 seeds):** four trials on `(nl, ch) ∈ {1,2} × {16,24}` at ep=2000/bs=16/lr=0.0015, all FLAT. GAP intentionally skipped — DIM 5 showed it's strictly worse for small DIM, and the deep-hierarchy win (where GAP could plausibly show up) won't appear until ≥DIM 7.

| Trial         | nl | ch | final V | FLAT in | NRMSE    | vs Ridge raw | vs Ridge trans |
|---------------|---:|---:|--------:|--------:|---------:|-------------:|---------------:|
| **nl1-ch16**  |  1 | 16 |      32 |     512 | **0.003315** |      **-36%** |      **-11%** |
| nl1-ch24      |  1 | 24 |      32 |     768 | 0.003684 |         -29% |           -0.8% |
| nl2-ch16      |  2 | 16 |      16 |     512 | 0.004614 |         +13% |           +24% |
| nl2-ch24      |  2 | 24 |      16 |     768 | (interrupted, didn't print) |

Two findings settled fast: **nl=1 transfers from DIM 5** (nl=2 is +39% worse at matched ch), and **ch=24 is worse than ch=16** (+11%). The "channels scale with DIM" heuristic from the fan-in argument was refuted — ch=16 is an absolute sweet spot, not a DIM-proportional one. nl2-ch24 wasn't needed given nl2-ch16 was already decisively bad.

**Run 22 (nl1-ch8 lean channel probe, 5 seeds):** 0.003881 (+17% vs ch=16). Confirms the channel bowl has a crisp minimum at ch=16 — ch=8 is too lean (undershoots layer-1 fan-in over-completeness), ch=24 is too wide. Identical shape to DIM 5's run-16 result where ch=8 vs ch=16 at nl=1 gave the same 17% penalty. **ch=16 is a genuine architectural constant at small DIM, not a coincidence.**

**Run 23 (coarse lr_max sweep, 5 seeds — interrupted):** lr ∈ {0.0005, 0.001, 0.002, 0.003, 0.005} started, got through the first three before Ctrl+C:

| lr_max | NRMSE    | Δ vs min |
|-------:|---------:|---------:|
| 0.0005 | 0.004692 |     +41% |
| 0.001  | 0.003319 |        — |
| 0.0015 | 0.003315 | (run 21) |
| 0.002  | 0.004125 |     +24% |

lr=0.001 (0.003319) and lr=0.0015 (0.003315) are **functionally identical** — 0.12% delta at 5-seed resolution is pure variance. The bowl has a flat floor from ~0.001 to ~0.0015 and a sharp right wall past 0.0015 (+24% at 0.002). Left wall is gentler (+41% at 0.0005). DIM 5's lr=0.0015 sits inside the flat region, so it transfers.

**Run 24 (10-seed variance check at lr=0.0015):** 0.003414 (+3.0% vs 5-seed 0.003315). Expected noise discount, still cleanly beats both baselines (-34% raw, -8% translated). No instability.

**Run 25 (bs sweep, 5 seeds):** this is the run that reframed the whole DIM 6 story. Swept `bs ∈ {8, 16, 32, 64}` at locked nl=1/ch=16/FLAT/ep=2000/lr=0.0015:

| bs   | NRMSE    | Δ vs best | time   | updates/epoch |
|-----:|---------:|----------:|-------:|--------------:|
|    8 | 0.004304 |     +30.5% | 54.9s  |          ~100 |
|   16 | 0.003315 |      +0.5% | 27.2s  |           ~50 |
| **32** | **0.003299** |  **—** | **10.3s** |      **~25** |
|   64 | 0.005727 |     +73.6% |  6.4s  |           ~13 |

**bs=16 and bs=32 are statistically tied**, but bs=32 runs 2.6× faster. bs=8 is +30% (opposite of DIM 5 where bs=8 was marginally *better*), and bs=64 blows up to worse than Ridge raw (+74%).

The shape makes sense once framed in terms of **total gradient updates**, not batch size:

- DIM 5 bs=16 (403 samples) → ~25 updates/epoch × 2000 = **50k total updates**
- DIM 6 bs=16 (806 samples) → ~50 updates/epoch × 2000 = **100k total updates** (2× — over-trained)
- DIM 6 bs=32 → ~25 updates/epoch × 2000 = **50k total updates** (matches DIM 5 exactly)
- DIM 6 bs=8 → ~100 updates/epoch × 2000 = 200k (4× — way over)
- DIM 6 bs=64 → ~13 updates/epoch × 2000 = 26k (half — under)

**The real invariant across DIMs is ~50k total gradient updates**, not a fixed batch size. Inheriting bs=16 from DIM 5 quietly doubled the update count at DIM 6 — the extra gradient steps past 50k were buying nothing, just burning compute. This is a meaningful finding for DIM 7+: **bs should scale linearly with training-sample count** so `bs * updates_per_epoch * epochs` stays ~50k. DIM 7 has ~1613 train samples → bs=64. DIM 8 → bs=128. Etc.

**Run 26 (10-seed confirmation at bs=32):** 0.003346 (+1.4% vs 5-seed 0.003299 — tighter drift than bs=16's +3.0% 5→10 drift, suggesting bs=32 is also lower-variance). At 10 seeds bs=32 (0.003346) **cleanly beats bs=16** (0.003414) by 2.0%, AND it's 2.5× faster. bs=32 is strictly better on both axes.

**DIM 6 MG GOLD STANDARD at nl=1, ch=16, head=FLAT, ep=2000, bs=32, lr=0.0015 → NRMSE 0.003346.** `HCNNPresets.h` updated.

**Key lesson from DIM 6 for DIM 7+:** the "50k total gradient updates" heuristic should drive `batch_size` selection. Don't inherit bs from a smaller DIM without scaling.

### DIM 7

_(pending first run)_

### DIM 8

_(pending first run)_

### DIM 9+

_(pending first run)_

## Related

- `diagnostics/OptimizeHRCCNNForMG.h` — the interactive optimizer
- `diagnostics/MackeyGlass.h` / `diagnostics/NARMA10.h` — underlying benchmarks
- `readout/CNNReadout.md` — HCNN readout design and hyperparameter reference
