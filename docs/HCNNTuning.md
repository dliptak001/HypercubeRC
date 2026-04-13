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

| DIM |   N   | Ridge raw | Ridge trans. | Best HCNN | layers | ch | ep | bs | lr_max | wd | seed | Notes |
|----:|------:|----------:|-------------:|----------:|-------:|---:|---:|---:|-------:|---:|-----:|:------|
|   5 |    32 |  0.006165 |     0.004375 |  0.004897 |      3 | 32 |2000|  16|  0.003 |  0 | 11822067163148543833 | **FINAL** (averaged, 3 CNN inits). -21% vs Ridge raw, +12% vs Ridge translated. nl=3 exceeds auto-rule cap (`min(DIM-3,4)`=2); real HCNNConv constraint is nl ≤ DIM-2. |
|   6 |    64 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|   7 |   128 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|   8 |   256 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|   9 |   512 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|  10 |  1024 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|  11 |  2048 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|  12 |  4096 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |

## NARMA-10

Data: 18·N collected samples, 70/30 train/test, input `u ~ U(0, 0.5)`,
target from the standard 10th-order recurrence.

| DIM |   N   | Ridge raw | Ridge trans. | Best HCNN | layers | ch | ep | bs | lr_max | wd | seed | Notes |
|----:|------:|----------:|-------------:|----------:|-------:|---:|---:|---:|-------:|---:|-----:|:------|
|   5 |    32 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|   6 |    64 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|   7 |   128 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|   8 |   256 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|   9 |   512 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|  10 |  1024 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|  11 |  2048 |         – |            – |         – |      – |  – |  – |  – |      – |  – |    – | not yet tuned |
|  12 |  4096 |         – |            – |         – |      – |  – |  – |  − |      – |  – |    – | not yet tuned |

## Column legend

- **layers**: Conv+Pool pairs (0 in `CNNReadoutConfig` means auto =
  `max(1, min(DIM-3, 4))` — this column records the *resolved* count)
- **ch**: base `conv_channels`; each successive layer doubles
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

### DIM 6

_(pending first run)_

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
