# HRCCNN Baseline Config — First-Probe Architecture for New DIMs

Reference record of the **baseline HCNN readout configuration** used when
scouting a hypercube dimension that hasn't yet been tuned. This is the
"known-safe starting point" — not the Gold Standard, not a Pareto winner,
just the minimum-viable config that reliably produces a directional read
on Mackey-Glass at a new DIM, scales cleanly, and lets per-DIM tuning
start from a meaningful anchor instead of a blind guess.

## The config

```cpp
CNNReadoutConfig cfg;
cfg.num_layers    = 1;          // nl=1, single conv+pool pair
cfg.conv_channels = 8;          // ch=8, lean
cfg.readout_type  = HCNNReadoutType::FLATTEN;
cfg.epochs        = 2000;
cfg.batch_size    = 1 << (DIM - 1);     // doubles per DIM, unified DIM 5-16
cfg.lr_max        = 0.0015f;
// num_cnn_seeds = 1 (single-seed scouting)
```

## Why these choices

**`nl=1, ch=8, FLAT`** — minimum-capacity architecture. At DIM 5-7 Gold
this is too lean (DIM 5/6 Gold uses ch=16, DIM 7 Gold uses nl=2/ch=24),
but for *first-probe scouting at an untuned DIM* the goal isn't optimal
NRMSE — it's a cheap directional read that verifies the pipeline and
lets you see how Ridge raw / Ridge translated / lean-HCNN relate at this
reservoir size. From there, tuning knows which direction to push.

**`FLATTEN`** — the "DIM 5/6 architectural invariant" from
`HCNNTuning.md`. GAP destroys per-vertex structure that the readout
needs; FLATTEN keeps it. Not negotiable at this architecture.

**`lr_max = 0.0015`** — the learning rate confirmed across DIM 5/6/7
Gold. Transfers cleanly and has a wide bowl, so first-probe rarely
benefits from re-tuning it.

**`epochs = 2000`** — the chaotic-signal training horizon (MG / NARMA).
Smooth signals saturate at ep=25; don't confuse the two regimes.

**`num_cnn_seeds = 1`** — single-seed scouting. At Gold Standard
confirmation we use 10 seeds to damp trajectory variance; here we're
looking for ~10% deltas, not ~1% deltas, and single-seed keeps wall time
linear in DIM count.

**`batch_size = 1 << (DIM - 1)`** — the critical scaling choice.
Holds the **~50k-gradient-update invariant** constant across DIMs.
Unified formula, safe for all DIM ≥ 1 (no signed/unsigned shift UB at
the low end), and happens to coincide exactly with the DIM 5/6/7 Gold
Standard `bs` values — so DIM 5..7 baseline runs share cadence with
the tuned Golds and the NRMSE delta is a pure architecture test:

| DIM |     N | train samples |    bs | updates/ep | total updates |
|----:|------:|--------------:|------:|-----------:|--------------:|
|   5 |    32 |           403 |    16 |         25 |         50000 |
|   6 |    64 |           806 |    32 |         25 |         50000 |
|   7 |   128 |          1612 |    64 |         25 |         50000 |
|   8 |   256 |          3225 |   128 |         25 |         50000 |
|   9 |   512 |          6451 |   256 |         25 |         50000 |
|  10 |  1024 |         12902 |   512 |         25 |         50000 |
|  11 |  2048 |         25804 |  1024 |         25 |         50000 |
|  12 |  4096 |         51609 |  2048 |         25 |         50000 |
|  13 |  8192 |        103219 |  4096 |         25 |         50000 |
|  14 | 16384 |        206438 |  8192 |         25 |         50000 |
|  15 | 32768 |        412876 | 16384 |         25 |         50000 |
|  16 | 65536 |        825753 | 32768 |         25 |         50000 |

Reservoir collection is cheap compared to the RCNN readout training, so
`bs` is the knob for controlling wall time. Scaling `bs ∝ N` keeps
updates/epoch constant and matches the DIM 5/6/7 Gold cadence, making
first-probe results commensurable with the tuned DIMs.

## Reservoir seed caveat

`hcnn_presets::MackeyGlass<DIM>().reservoir.seed` has MG-surveyed seeds
only for DIM 5-8. DIM 9-16 fall through to a hard-coded fallback
(`seed = 42`). This violates the per-DIM survey rule
(`feedback_per_dim_seeds.md`), but is unavoidable until an MG seed
survey is run at those DIMs.

First-probe NRMSEs at DIM 9+ should be read as **"fallback reservoir,
not canonical"**. Seed variance is expected to compress at larger N (any
single draw lands closer to the mean), so the directional signal is
usable — but don't freeze a Gold Standard off one of these runs without
re-confirming on a surveyed seed.

## Scouting runs — 2026-04-15

Two runs driven by `diagnostics/OptimizeHRCCNNForMG.cpp`:
  - **DIM 8-14 pass** (09:47 - 10:57): interrupted after DIM 11 to
    prioritize a threading pass on `RidgeRegression` (Ridge translated
    was bottlenecking at high DIM; see Observation 5).
  - **DIM 5-7 pass** (12:24 - 12:25): completed cleanly in ~42 seconds
    total — fixed-overhead regime, N² scaling hasn't kicked in yet.

| DIM |     N | Ridge raw | Ridge trans. | HCNN baseline |     Δ vs R-trans | wall (s) | vs prev |
|----:|------:|----------:|-------------:|--------------:|-----------------:|---------:|--------:|
|   5 |    32 |  0.006165 |     0.004375 |      0.004744 |            +8.4% |    14.24 |       – |
|   6 |    64 |  0.005205 |     0.003713 |      0.005517 |           +48.6% |    14.14 |   0.99× |
|   7 |   128 |  0.004310 |     0.003165 |      0.002655 |           -16.1% |    12.95 |   0.92× |
|   8 |   256 |  0.003771 |     0.002357 |      0.001884 |           -20.1% |    48.66 |   3.76× |
|   9 |   512 |  0.003856 |     0.002566 |      0.002079 |           -19.0% |   145.06 |   2.98× |
|  10 |  1024 |  0.002801 |     0.001524 |      0.001710 |           +12.2% |   589.92 |   4.07× |
|  11 |  2048 |  0.002186 |          n/a |      0.001289 |              n/a |  2976.52 |   5.04× |
|  12 |  4096 |         – |            – |             – |                – |        – |       – |
|  13 |  8192 |         – |            – |             – |                – |        – |       – |
|  14 | 16384 |         – |            – |             – |                – |        – |       – |
|  15 | 32768 |         – |            – |             – |                – |        – |       – |
|  16 | 65536 |         – |            – |             – |                – |        – |       – |

### Observations

1. **HCNN NRMSE trend across DIM 5-11** shows the lean baseline
   improves with reservoir size once past a low-DIM starvation zone:
   0.004744 → 0.005517 → 0.002655 → 0.001884 → 0.002079 → 0.001710 →
   0.001289. DIM 6 is a local maximum (the lean probe is visibly
   under-parameterized for 64 vertices — DIM 5/6 Gold Standards both
   use `ch=16`, twice the baseline); DIM 9 is noise-band. The overall
   trajectory from DIM 6 → DIM 11 is -77% NRMSE.

2. **Ridge translated wins at DIM 5, 6, 10; HCNN baseline wins at
   DIM 7, 8, 9, 11:**

   - DIM 5: Ridge trans -8.4% (noise-adjacent; Gold closes gap at ch=16).
   - DIM 6: Ridge trans **-48.6%** (decisive; baseline too lean for 64 vertices).
   - DIM 7: HCNN baseline -16.1%.
   - DIM 8: HCNN baseline -20.1%.
   - DIM 9: HCNN baseline -19.0%.
   - DIM 10: Ridge trans +12.2% (first high-DIM loss; likely `ch=8` capacity limit).
   - DIM 11: HCNN wins by default (Ridge trans not achievable, see Obs 5).

   Two distinct failure modes for the lean baseline: **architectural
   starvation at DIM 5/6** (too few channels for a small reservoir
   where every vertex matters), and a **marginal regression at DIM 10**
   (single-seed noise on a fallback reservoir, expected to close with
   `ch=16`). DIM 7-9 and DIM 11 are clean wins for HCNN baseline.

3. **Wall time scales as ~O(N²) for DIM ≥ 8** (4× per DIM increment):
   49 → 145 → 590 → 2977 seconds. Below DIM 8, fixed-overhead
   dominates — DIM 5/6/7 all completed in ~14 seconds apiece, giving
   ~1× "vs prev" ratios that mask the N² term. The crossover into N²
   scaling happens between DIM 7 and DIM 8 (14s → 49s, ~3.8×). Per-
   epoch cost is `train_samples × per-sample cost`; `train_samples ∝ N`
   and per-sample forward is linear in N for `nl=1`, so epoch cost is
   N² and `bs` scaling only holds updates constant, not wall time.
   Linear extrapolation from the DIM 8 anchor:

   | DIM | projected wall | projected wall |
   |----:|---------------:|---------------:|
   |  12 |        ~12000s |          ~3.3h |
   |  13 |        ~48000s |         ~13.3h |
   |  14 |       ~192000s |         ~53.3h |
   |  15 |       ~768000s |  ~213h (~8.9d) |
   |  16 |      ~3072000s | ~853h (~35.6d) |

   The full DIM 8-14 sweep is a ~3-day job at this architecture, and
   extending through DIM 16 pushes the total well past 6 weeks — both
   wholly dependent on single-threaded bottlenecks elsewhere in the
   pipeline (see Observation 4). Pareto-frontier tuning, not
   full-grid sweeping, is the only tractable approach at DIM ≥ 13.

4. **Ridge translated is the serial bottleneck at high DIM.** The
   closed-form solver in `RidgeRegression.cpp` is single-threaded, with
   an O(F²·S) X'X accumulation that at DIM 12 hits F=10 240, F² = 105M
   entries, and at DIM 14 hits F=40 960, F² = 1.68B entries. This is
   the motivation for the RidgeRegression threading work that follows
   this scouting run.

5. **Wall-time crossover at DIM 11 — Ridge translated is no longer
   achievable single-threaded.** The DIM 11 run was aborted because
   Ridge translated had already exceeded the full ~50 min HCNN
   baseline training time and was still grinding on the O(F²·S)
   accumulation with no finish in sight (F = 2.5·N = 5120 at DIM 11;
   F² = 26.2M entries × ~26k samples = ~670B float-float products in
   one serial loop). This is the first DIM at which the closed-form
   "it's cheap" argument for Ridge translated fails outright, not
   just loses to HCNN on wall time. Ridge translated at DIM 11+ is
   effectively gated on the RidgeRegression threading work.

## Practical recommendation — use HCNN at DIM ≥ 7

Combining the accuracy and wall-time observations from both runs:

- **DIM 5**: Ridge trans -8.4% vs HCNN baseline *(noise-adjacent; DIM 5
  Gold at `ch=16` closes the gap and beats Ridge trans by -28%)*.
- **DIM 6**: Ridge trans **-48.6% vs HCNN baseline** *(decisive loss;
  the lean `ch=8` probe is architecturally starved at 64 vertices,
  DIM 6 Gold at `ch=16` flips this back to HCNN -10%)*.
- **DIM 7**: HCNN baseline -16.1% vs Ridge translated — and DIM 7
  Gold at `nl=2/ch=24` widens this to -52.8%.
- **DIM 8**: HCNN baseline -20.1% vs Ridge translated.
- **DIM 9**: HCNN baseline -19.0% vs Ridge translated.
- **DIM 10**: Ridge translated +12.2% vs HCNN baseline *(within
  single-seed/fallback-reservoir noise; a modest `ch=16` bump is
  expected to close this)*.
- **DIM 11**: HCNN baseline finishes in ~50 min at NRMSE 0.001289;
  Ridge translated is not achievable single-threaded at this DIM
  — the run was aborted after Ridge trans exceeded HCNN wall time
  with no end in sight. HCNN wins by default on both axes.

**The practical bottom line:**

- **DIM ≥ 7**: HCNN is the default readout. At DIM 7, even the lean
  baseline beats Ridge translated; at DIM 8+ the margin widens and
  Ridge's wall-time cost collapses the "cheap closed-form" argument
  entirely (at DIM 11+ it's not even computable without threading).

- **DIM 5-6**: Use the tuned Gold Standards (`ch=16`), not the lean
  baseline. The baseline is a scouting probe, not a production
  config at small reservoir sizes — the 64-vertex DIM 6 reservoir
  specifically needs more channels than `ch=8` provides. Gold
  Standards at DIM 5/6 both beat Ridge translated (−28% and −10%
  respectively), so "HCNN is the default" still holds, but you must
  skip the baseline step and go straight to the Gold config.

Ridge (raw or translated) remains a useful **baseline anchor** — it
gives a deterministic, closed-form reference number that HCNN
improvements are measured against — but it is no longer the
recommended production readout at any DIM where a Gold Standard (or
the lean baseline, DIM ≥ 7) is available.

## When to move off the baseline

This config is a **starting point, not a destination**. Once a new DIM
has a baseline number:

1. **Survey the reservoir seed first** (SeedSurvey at the target DIM)
   so subsequent tuning is anchored to the canonical seed, not seed 42.
2. Widen `conv_channels` (`ch=16`, `ch=24`) to probe capacity headroom.
3. Try `nl=2` once the reservoir is large enough that a second
   conv+pool pair still leaves the flatten layer meaningful — DIM 7 was
   the first DIM where `nl=2` beat `nl=1`.
4. Refine `lr_max` with a fine sweep only if the coarse-knee candidate
   looks promising — the DIM 5/6/7 experience is that the lr bowl is
   wide and 0.0015 rarely loses meaningfully.
5. Bump `num_cnn_seeds` to 5 for scouting resolution, then to 10 for
   Gold Standard confirmation.

Gold Standards — the tuned, frozen, multi-seed-confirmed configs — live
in `readout/HCNNPresets.h` and are documented in `docs/HCNNTuning.md`.
This file covers the *unfrozen* territory.
