# Examples

## BasicPrediction

The minimal hello-world for HypercubeRC. Demonstrates the complete pipeline on a
sine wave: create a reservoir, drive it with input, extract stride-selected output
features, train a Ridge readout, and evaluate prediction quality.

This is the place to start if you want to understand how the pieces fit together.

**What it shows:**
- ESN construction with a single DIM template parameter
- Warmup (wash out initial transients) and Run (collect states)
- Output selection via `output_fraction` (10% of vertices by default)
- Raw vs translation feature extraction (command-line selectable)
- Ridge readout training and evaluation (R², NRMSE)
- Sample-by-sample prediction output

**Expected output (abbreviated):**
```
=== HypercubeRC: Basic Sine Wave Prediction ===

Config: DIM=7  N=128  Outputs=15 (10%)  Features=15 (raw)  Readout=Ridge  Horizon=1

--- Results (test set: 600 samples) ---
  R2:    ~1.000000   (effectively perfect)
  NRMSE: ~0.000xxx   (sub-0.1% error)

--- Sample predictions ---
  Step  |   Actual   |  Predicted  |    Error
   1400 |  +0.xxxxx  |   +0.xxxxx  |  +0.00xxx
   ...
```

**Make it yours:** Replace the sine wave generation (lines 58-60) with your own time
series data. Keep values in [-1, 1] (the reservoir clamps out-of-range inputs silently).
Adjust `DIM` to control reservoir size, and `warmup`/`collect` to match your data volume.

## SignalClassification

Classify four waveform types — sine, square, triangle, chirp — from reservoir state
alone. One-vs-rest Ridge readouts produce a confusion matrix and transition dynamics
analysis showing how quickly the reservoir locks on after a waveform switch.

**What it shows:**
- Reservoir as a feature extractor for pattern recognition
- One-vs-rest classification with argmax over readout scores
- Translation features (2.5N) for improved class separability
- Confusion matrix and per-class accuracy breakdown
- Transition dynamics: accuracy vs steps after waveform switch

**Expected output (abbreviated):**
```
=== HypercubeRC: Signal Classification ===

Config: DIM=7  N=128  Outputs=90 (70%)  Features=225 (translation)  Readout=Ridge

Overall accuracy: ~96.8%

Per-class breakdown:
  Sine       100%  -- perfectly separable
  Square     100%  -- perfectly separable
  Triangle   ~99%  -- near-perfect
  Chirp      ~88%  -- hardest class

  Steps after switch  | Accuracy
  0 - 3               |  ~75%
  Entire block        |  ~97%
```

**Make it yours:** Add your own waveform types to `GenerateWaveform()` and increase
`NUM_CLASSES`. Adjust `block_size` to match your expected signal duration.

## StreamingAnomaly

Simulates industrial process monitoring. The reservoir learns normal process behavior
during a priming phase, then monitors a live stream in 200-step windows. Three anomaly
types are injected — noise spike, DC drift, and frequency shift — each for 3 windows,
separated by normal operation to show both detection and recovery.

**What it shows:**
- Batch training on historical "normal" data (priming)
- Anomaly detection via prediction error exceeding a threshold (5x baseline RMSE)
- Three distinct anomaly signatures with different RMSE ratios
- Automatic recovery without retraining as anomalies end
- Effect of leak rate on detection sensitivity vs recovery speed

**Expected output (abbreviated):**
```
=== HypercubeRC: Streaming Anomaly Detection ===

Config: DIM=8  N=256  Outputs=128 (50%)  Features=128 (raw)  Readout=Ridge

--- Phase 1: Learn what "normal" looks like ---
  Baseline RMSE: ~0.0065

--- Phase 2: Monitor (30 windows of 200 steps) ---
  Window | Condition   |     RMSE     | Ratio | Status
      1  | Normal      |     ~0.006   |  ~1.0 |
      6  | Noise spike |     ~0.07    | ~12.0 | ** ANOMALY **
     14  | DC drift    |     ~0.43    | ~67.0 | ** ANOMALY **
     22  | Freq shift  |     ~0.16    | ~25.0 | ** ANOMALY **
```

**Make it yours:** Replace `GenerateProcess()` with your real sensor data feed.
Adjust `normal_noise` to match your signal characteristics. Tune the
`anomaly_threshold` multiplier (5x is conservative; 3x catches subtler changes).

## Building

All three examples build automatically alongside the main benchmark:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/BasicPrediction
./build/SignalClassification
./build/StreamingAnomaly
```

In CLion, select the target from the run configuration dropdown (top toolbar).
