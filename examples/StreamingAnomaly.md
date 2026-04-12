# Streaming Anomaly Detection — Industrial Process Monitoring

## What this example demonstrates

A reservoir learns what "normal" looks like for a simulated industrial
process, then monitors a live stream for deviations. Three distinct
anomaly types are injected — a noise spike, a DC drift, and a frequency
shift — each separated by normal operation. The example shows clean
detection of all three and automatic recovery without retraining.

**Two readouts run side-by-side** on the same reservoir dynamics for
apples-to-apples comparison:

- **Ridge** — closed-form regression on stride-selected features (50%
  output fraction). Cheap to train; Ridge is the natural fit when you
  need to re-prime the model frequently.
- **HCNN** — learned CNN on raw state (all vertices). Trained once in
  Phase 1 and frozen for Phase 2. HCNN does **not** support incremental
  updates — `CNNReadout` is batch only — so the example uses it in
  frozen-readout mode.

This is the most practical example in the collection: it demonstrates
how a frozen reservoir model can serve as a real-time process monitor,
and lets you compare how two very different readout strategies produce
the same kind of anomaly signal from identical reservoir state.

## Conceptual background

The core idea: if a model can predict normal behavior accurately, then
a sudden increase in prediction error signals that something has changed.
The reservoir is trained to predict the next value of a multi-harmonic
process signal. During monitoring, each window's RMSE is compared to the
baseline error from the training phase. If the ratio exceeds a threshold
(default 5x), an anomaly is flagged.

This approach has several appealing properties:

- **No anomaly labels needed.** The model only learns "normal" — it doesn't
  need examples of every possible failure mode.
- **Automatic recovery.** Once the anomaly ends, the reservoir state washes
  out and prediction error returns to baseline. No retraining required.
- **Interpretable signal.** The RMSE ratio directly indicates how different
  the current behavior is from normal. A ratio of 12x is a much stronger
  anomaly than 2x.

## The simulated process

The "normal" process is a multi-harmonic signal:

```
signal(t) = 0.6 * sin(0.1t) + 0.2 * sin(0.3t) + noise
```

where noise is uniform in [-0.01, +0.01]. Three anomaly types are injected:

| Anomaly | What changes | Magnitude | Physical analogy |
|---------|-------------|-----------|------------------|
| Noise spike | Noise level 0.01 -> 0.12 | 12x noise | Sensor degradation |
| DC drift | +0.30 offset added | Systematic bias | Sensor fouling |
| Freq shift | Frequency multiplied by 1.3 | Dynamics change | Motor speed change |

## The pipeline

```
Phase 1: Learn normal                         Phase 2: Monitor

                       ┌─> Ridge train ┐                        ┌─> Ridge RMSE ─┐
Normal signal ──> ESN ─┤               ├─>  Live signal ──> ESN─┤               ├─> > 5x baseline?
  4000 steps           └─> HCNN train ─┘      200-step           └─> HCNN  RMSE ─┘       ALERT
                       (both frozen after)    windows            (frozen readouts)
```

**Phase 1 — Learn normal behavior:**

1. Generate 4000 steps of the normal process signal.
2. Warmup two reservoirs (500 steps each, same seed/config), then
   collect states (4000 steps) from both.
3. Train a Ridge readout and an HCNN readout to predict the next value
   from reservoir state. Both see the same targets.
4. Measure baseline RMSE per readout on a held-out portion of the
   training data (their baselines differ because the readouts differ).
5. Set each readout's anomaly threshold at 5x its own baseline.

**Phase 2 — Stream monitoring:**

1. Process the signal in 200-step windows (30 windows total).
2. For each window: run both reservoirs, predict with each frozen
   readout, compute RMSE for each.
3. If a readout's window RMSE exceeds its threshold, flag that readout
   for this window. The status column prints `R`, `H`, or `R+H`
   depending on which readouts flagged.

The reservoir state is cleared between windows via `ClearStates()`, but
the reservoir's internal neuron states carry over — this is what allows
the model to detect when dynamics have changed and to recover gradually
when they return to normal.

## What to expect

### Leak rate = 0.3 (leaky integrator, default), Ridge + HCNN

DIM=8, 256 neurons, Ridge at 50% output (128 vertices) + HCNN at 100%
output (all 256 vertices) on the same reservoir dynamics, leak_rate=0.3.
Both readouts trained once in Phase 1; Phase 2 is frozen prediction.

Baselines (prime test RMSE):
- Ridge: ~0.0068   threshold ~0.034
- HCNN : ~0.0064   threshold ~0.032

```
                              Ridge RMSE /  ratio  |  HCNN RMSE / ratio  | Status
Windows  1-5:   Normal        ~0.0067     /  ~1.0  |  ~0.0065   / ~1.0   |
Windows  6-8:   Noise spike   ~0.080      / ~11-12 |  ~0.075    / ~11-12 | R+H
Window   9:     Normal        ~0.0067     /  ~1.0  |  ~0.0063   / ~1.0   | (instant recovery)
Windows 10-13:  Normal        ~0.0068     /  ~1.0  |  ~0.0064   / ~1.0   |
Windows 14-16:  DC drift      ~0.48-0.56  / ~71-82 |  ~0.37-0.43/ ~57-66 | R+H
Window  17:     Normal        ~0.17       / ~25    |  ~0.069    / ~11    | R+H (slow washout)
Windows 18-21:  Normal        ~0.0068     /  ~1.0  |  ~0.0065   / ~1.0   |
Windows 22-24:  Freq shift    ~0.13-0.21  / ~19-31 |  ~0.11-0.17/ ~17-27 | R+H
Window  25:     Normal        ~0.16       / ~24    |  ~0.12     / ~19    | R+H (slow washout)
Windows 26-30:  Normal        ~0.0068     /  ~1.0  |  ~0.0065   / ~1.0   |

Flagged windows: Ridge=11, HCNN=11  (9 anomaly + 2 washout)
```

**What to notice:**

- **Same detection coverage.** Both readouts flag exactly the same 11
  windows — the 9 true-anomaly windows plus the two slow-washout
  windows (17 and 25) where the leaky integrator is still ringing
  from the prior anomaly.
- **HCNN is slightly tighter on normal baselines.** HCNN's baseline
  RMSE is ~5% lower than Ridge's, so its ratios during normal
  operation sit just below 1.0.
- **HCNN is more conservative on DC drift.** Ridge spikes to 71-82x,
  HCNN to 57-66x. Both are well above threshold — the conservative
  HCNN ratio is not a miss, just a cleaner signal.
- **HCNN recovers faster after DC drift.** Window 17: Ridge still at
  25x (slow washout), HCNN already at 11x. By window 18 both are
  back to baseline.
- **Training cost.** Ridge priming is ~0.02s (closed-form solve).
  HCNN priming is ~100s at 100 epochs on a DIM=8 reservoir — two
  orders of magnitude slower. This is the real tradeoff: HCNN is
  batch-only and expensive to re-prime, so it fits "prime once,
  monitor forever" workloads much better than scenarios where you
  need to frequently retrain on fresh normal data.

### Leak rate = 1.0 (full replacement)

Same configuration with leak_rate=1.0:

```
Windows  1-5:   Normal         RMSE ~0.008   ratio ~1.0
Windows  6-8:   Noise spike    RMSE ~0.09    ratio ~12x  ** ANOMALY **
Window   9:     Normal         RMSE ~0.008   ratio ~1.1  (instant recovery)
Windows 10-13:  Normal         RMSE ~0.008   ratio ~1.0
Windows 14-16:  DC drift       RMSE ~0.051   ratio ~6.5  ** ANOMALY **
Window  17:     Normal         RMSE ~0.020   ratio ~2.6  (1-window washout)
Windows 18-21:  Normal         RMSE ~0.008   ratio ~1.0
Windows 22-24:  Freq shift     RMSE ~0.06-0.08  ratio ~8-10x  ** ANOMALY **
Window  25:     Normal         RMSE ~0.042   ratio ~5.4  (slow washout)
Windows 26-30:  Normal         RMSE ~0.008   ratio ~1.0
```

### Comparison

| Anomaly | Ratio @ leak=0.3 | Ratio @ leak=1.0 | Change |
|---------|-------------------|-------------------|--------|
| Noise spike | ~12x | ~12x | Same |
| DC drift | ~67x | ~6.5x | 10x more sensitive |
| Freq shift | ~25-35x | ~8-10x | 3-4x more sensitive |
| Baseline RMSE | 0.0065 | 0.0078 | 17% lower |

### Effect of leak rate on anomaly detection

The leaky integrator (leak_rate < 1.0) causes neurons to retain a
fraction of their previous state at each step. This has two effects
on anomaly detection:

**Higher sensitivity to sustained anomalies.** With leak=1.0, a DC
offset only affects the current step's activation — the neuron fully
replaces its state. With leak=0.3, 70% of the old (offset-contaminated)
state persists, so the error compounds across steps. DC drift jumps
from 6.5x to 67x, and frequency shifts from ~8x to ~30x.

**Slower recovery after anomalies.** The sticky state takes longer to
wash out. With leak=1.0, the first normal window after a freq shift
shows 5.4x (borderline). With leak=0.3, it shows 23x — the slow
neurons are still ringing from the altered dynamics. Recovery takes
one extra window.

**Noise spike is unaffected.** Random noise averages out regardless
of leak rate. The leaky integrator smooths it slightly (baseline RMSE
drops 17%) but the anomaly ratio stays at ~12x.

**For anomaly detection, the slower recovery is a feature, not a bug.**
You want the alarm to persist until the system fully stabilizes. A
leak rate of 0.3 provides dramatically higher sensitivity to the
anomaly types that matter most in industrial monitoring (systematic
drift and dynamics changes) while adding only one extra window of
washout time.

## Things to try

- **Leak rate.** Set `base_cfg.leak_rate` in the source. The default
  is 0.3. Try 1.0 for fast recovery with lower sensitivity, or 0.1 for
  even higher sensitivity with slower washout. Both readouts see the
  same reservoir, so the sensitivity change applies to both.

- **Ridge output fraction.** Set `ridge_cfg.output_fraction` in the
  source. The default is 0.5 (128 of 256 vertices). HCNN always uses
  all vertices — its `output_fraction` is fixed at 1.0.

- **HCNN epochs.** The default is `cnn_cfg.epochs = 25`. On this
  smooth multi-harmonic process signal HCNN saturates very early —
  25 epochs matches 100+ epochs within measurement noise for every
  anomaly window. Raise to 50-100 if you change the signal or the
  reservoir config and want more headroom. Keep `lr_max <= 0.003`
  to avoid the denormal/NaN slow path.

- **Lower the threshold.** Change `anomaly_threshold` from 5.0 to 2.0.
  You'll catch anomalies sooner but may see false positives during
  washout windows.

- **Translation features (Ridge only).** Pass `translation` as a
  command-line argument. The 2.5M feature set may change Ridge's
  detection sensitivity. The HCNN path always uses raw state.

- **Window size.** Smaller windows (e.g., 50 steps) make detection faster
  but noisier. Larger windows (e.g., 500) smooth the RMSE estimate but
  delay detection.

## A note on streaming and HCNN

`CNNReadout` is **batch-only**. It does not expose a `TrainIncremental`
method, and there is no cheap way to blend new training data into an
existing network while preserving Adam's optimizer state. For
applications where the model must adapt online to slow drift, use
`LinearReadout` with `TrainIncremental` — see the streaming workflow
section of `docs/Readout.md`.

HCNN is included here to show that its **frozen** prediction error is
also a usable anomaly signal, and to make it easy to compare the two
readouts' sensitivity profiles on identical reservoir dynamics. If you
don't need that comparison and your workload genuinely needs to track
drift, drop the HCNN path and stick with LinearReadout.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/StreamingAnomaly              # default: raw features
./build/StreamingAnomaly raw          # explicit raw
./build/StreamingAnomaly translation  # translation 2.5M features
```
