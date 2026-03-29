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

With DIM=8, raw features, Ridge readout:

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

**Recovery dynamics** are the most interesting part:

- **Noise spike** — instant recovery (window 9 is back to baseline).
  Random noise doesn't alter the reservoir's internal dynamics.

- **DC drift** — 1-window washout. The offset shifts the reservoir
  state away from the normal trajectory; it takes one window for the
  tanh neurons to re-center.

- **Frequency shift** — slowest recovery (1-2 extra windows). Changed
  dynamics alter the reservoir's internal oscillation patterns, which
  take longer to wash out than a simple offset.

## Things to try

- **Lower the threshold.** Change `anomaly_threshold` from 5.0 to 2.0.
  You'll catch anomalies sooner but may see false positives during
  washout windows.

- **Translation features.** Pass `translation` as a command-line argument.
  The 2.5N feature set may change detection sensitivity.

- **Window size.** Smaller windows (e.g., 50 steps) make detection faster
  but noisier. Larger windows (e.g., 500) smooth the RMSE estimate but
  delay detection.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/StreamingAnomaly              # default: raw features
./build/StreamingAnomaly raw          # explicit raw
./build/StreamingAnomaly translation  # translation 2.5N features
```
