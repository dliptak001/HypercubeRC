# Streaming Anomaly Detection — Industrial Process Monitoring

## What this example demonstrates

A reservoir learns what "normal" looks like for a simulated industrial
process, then monitors a live stream for deviations. Three distinct
anomaly types are injected — a noise spike, a DC drift, and a frequency
shift — each separated by normal operation. The example shows clean
detection of all three and automatic recovery without retraining.

The HCNN readout is trained once in Phase 1 and frozen for Phase 2.
Prediction error is the anomaly signal.

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
Phase 1: Learn normal                    Phase 2: Monitor

Normal signal ──> ESN ──> HCNN train     Live signal ──> ESN ──> HCNN RMSE ──> > 5x baseline?
  4000 steps              (frozen after)   200-step               (frozen)          ALERT
                                           windows
```

**Phase 1 — Learn normal behavior:**

1. Generate 4000 steps of the normal process signal.
2. Warmup the reservoir (500 steps), then collect states (4000 steps).
3. Train the HCNN readout to predict the next value from reservoir state.
4. Measure baseline RMSE on a held-out portion of the training data.
5. Set anomaly threshold at 5x baseline.

**Phase 2 — Stream monitoring:**

1. Process the signal in 200-step windows (30 windows total).
2. For each window: run the reservoir, predict with the frozen readout,
   compute RMSE.
3. If the window RMSE exceeds the threshold, flag as anomaly.

Between windows, `ClearStates()` clears the collected output buffer
(so the readout can index the new window's timesteps from zero). The
reservoir neurons' live activations are not reset — they carry over,
which is what allows the model to detect when dynamics have changed
and to recover gradually when they return to normal.

## What to expect

### Leak rate = 0.3 (leaky integrator)

DIM=8, 256 neurons, readout on all 256 vertices, leak_rate=0.3.
Trained once in Phase 1; Phase 2 is frozen prediction.

Baseline RMSE: ~0.0064, threshold ~0.032.

```
  Window | Condition          |    RMSE     Ratio | Status
  -------+--------------------+-------------------+---------
  1-5      Normal               ~0.0065    ~1.0
  6-8      Noise spike          ~0.075     ~12      ** ANOMALY **
  9        Normal               ~0.0065    ~1.0     (instant recovery)
  10-13    Normal               ~0.0065    ~1.0
  14-16    DC drift             ~0.58-0.62 ~92-98   ** ANOMALY **
  17       Normal               ~0.15      ~23      ** ANOMALY ** (slow washout)
  18-21    Normal               ~0.0065    ~1.0
  22-24    Freq shift           ~0.22-0.32 ~35-50   ** ANOMALY **
  25       Normal               ~0.23      ~35      ** ANOMALY ** (slow washout)
  26-30    Normal               ~0.0065    ~1.0

Flagged windows: 11  (9 anomaly + 2 washout)
```

**What to notice:**

- **Clean detection of all three anomaly types.** Every anomaly window
  is flagged with ratios well above the 5x threshold.
- **Instant recovery from noise spike.** Random noise has no persistent
  effect on reservoir state — the next normal window is back to baseline.
- **Slow washout after DC drift and freq shift.** The leaky integrator
  compounds the offset/dynamics change across steps, producing very high
  ratios (~92-98x for drift, ~35-50x for freq shift). The flip side:
  it takes 1-2 extra windows to wash out after the anomaly ends.
- **The 2 washout windows are features, not false positives.** They
  confirm the reservoir hasn't fully stabilized yet — exactly what you
  want an alarm to signal.

### Leak rate = 1.0 (full replacement)

Same configuration with leak_rate=1.0:

| Anomaly | Ratio @ leak=0.3 | Ratio @ leak=1.0 |
|---------|-------------------|-------------------|
| Noise spike | ~12x | ~12x |
| DC drift | ~92-98x | ~6.5x |
| Freq shift | ~35-50x | ~8-10x |
| Baseline RMSE | 0.0064 | 0.0078 |

### Effect of leak rate on anomaly detection

The leaky integrator (leak_rate < 1.0) causes neurons to retain a
fraction of their previous state at each step. This has two effects
on anomaly detection:

**Higher sensitivity to sustained anomalies.** With leak=1.0, a DC
offset only affects the current step's activation — the neuron fully
replaces its state. With leak=0.3, 70% of the old (offset-contaminated)
state persists, so the error compounds across steps. DC drift jumps
from 6.5x to ~92-98x, and frequency shifts from ~8x to ~35-50x.

**Slower recovery after anomalies.** The sticky state takes longer to
wash out. With leak=1.0, the first normal window after a freq shift
shows 5.4x (borderline). With leak=0.3, it shows ~35x — the slow
neurons are still ringing from the altered dynamics.

**Noise spike is unaffected.** Random noise averages out regardless
of leak rate. The anomaly ratio stays at ~12x.

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

- **HCNN epochs.** The default is 1000. On this smooth multi-harmonic
  process signal HCNN saturates early — try lowering to 100 or 25 to
  verify. Keep `lr_max <= 0.005` to avoid denormal/NaN.

- **Lower the threshold.** Change `anomaly_threshold` from 5.0 to 2.0.
  You'll catch anomalies sooner but may see false positives during
  washout windows.

- **Window size.** Smaller windows (e.g., 50 steps) make detection faster
  but noisier. Larger windows (e.g., 500) smooth the RMSE estimate but
  delay detection.

## A note on online training

`Readout` also supports online training for streaming applications.
For workloads that need to track drift, HCNN online training can adapt
the model incrementally — see `docs/Readout.md` for details. This
example uses frozen-readout mode to demonstrate pure anomaly detection.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/StreamingAnomaly
```
