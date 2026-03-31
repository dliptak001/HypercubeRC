# Streaming Anomaly Detection — Industrial Process Monitoring

## What this example demonstrates

A reservoir learns what "normal" looks like for a simulated industrial
process, then monitors a live stream for deviations. Three distinct
anomaly types are injected — a noise spike, a DC drift, and a frequency
shift — each separated by normal operation. The example shows clean
detection of all three and automatic recovery without retraining.

This is the most practical example in the collection: it demonstrates
how a frozen reservoir model can serve as a real-time process monitor.

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
Phase 1: Learn normal                Phase 2: Monitor

Normal signal ──> Reservoir ──>      Live signal ──> Reservoir ──>
  4000 steps       Train readout       200-step        Predict ──> RMSE
                   Measure baseline     windows                    vs baseline
                                                                   > 5x? ALERT
```

**Phase 1 — Learn normal behavior:**

1. Generate 4000 steps of the normal process signal.
2. Warmup the reservoir (500 steps), then collect states (4000 steps).
3. Train a Ridge readout to predict the next value from reservoir state.
4. Measure baseline RMSE on a held-out portion of the training data.
5. Set the anomaly threshold at 5x baseline.

**Phase 2 — Stream monitoring:**

1. Process the signal in 200-step windows (30 windows total).
2. For each window: run the reservoir, predict with the frozen readout,
   compute RMSE.
3. If RMSE > threshold, flag as anomaly.

The reservoir state is cleared between windows via `ClearStates()`, but
the reservoir's internal neuron states carry over — this is what allows
the model to detect when dynamics have changed and to recover gradually
when they return to normal.

## What to expect

### Leak rate = 0.3 (leaky integrator, default)

DIM=8, 256 neurons, 50% output (128 vertices), raw features, Ridge readout,
leak_rate=0.3:

```
Windows  1-5:   Normal         RMSE ~0.006   ratio ~1.0
Windows  6-8:   Noise spike    RMSE ~0.07    ratio ~12x  ** ANOMALY **
Window   9:     Normal         RMSE ~0.007   ratio ~1.1  (instant recovery)
Windows 10-13:  Normal         RMSE ~0.007   ratio ~1.0
Windows 14-16:  DC drift       RMSE ~0.43    ratio ~67x  ** ANOMALY **
Window  17:     Normal         RMSE ~0.108   ratio ~17x  ** ANOMALY ** (slow washout)
Windows 18-21:  Normal         RMSE ~0.006   ratio ~1.0
Windows 22-24:  Freq shift     RMSE ~0.16-0.23  ratio ~25-35x  ** ANOMALY **
Window  25:     Normal         RMSE ~0.150   ratio ~23x  ** ANOMALY ** (slow washout)
Windows 26-30:  Normal         RMSE ~0.006   ratio ~1.0
```

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

- **Leak rate.** Set `cfg.leak_rate` in the source. The default is 0.3.
  Try 1.0 for fast recovery with lower sensitivity, or 0.1 for even
  higher sensitivity with slower washout.

- **Output fraction.** Set `cfg.output_fraction` in the source. The default
  is 0.5 (128 of 256 vertices). Try 1.0 for full output or lower values
  to reduce Ridge readout cost.

- **Lower the threshold.** Change `anomaly_threshold` from 5.0 to 2.0.
  You'll catch anomalies sooner but may see false positives during
  washout windows.

- **Translation features.** Pass `translation` as a command-line argument.
  The 2.5M feature set may change detection sensitivity.

- **Window size.** Smaller windows (e.g., 50 steps) make detection faster
  but noisier. Larger windows (e.g., 500) smooth the RMSE estimate but
  delay detection.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/StreamingAnomaly              # default: raw features
./build/StreamingAnomaly raw          # explicit raw
./build/StreamingAnomaly translation  # translation 2.5M features
```
