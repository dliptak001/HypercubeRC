# Examples

## BasicPrediction

The minimal hello-world for HypercubeRC. Demonstrates the complete pipeline on a
sine wave with **two readouts** side-by-side: Ridge regression on stride-selected
features (fast, closed-form) and HypercubeCNN on raw state (learned convolutional
readout). Both run on the same reservoir dynamics for an apples-to-apples comparison.

This is the place to start if you want to understand how the pieces fit together.

**What it shows:**
- ESN construction with a single DIM template parameter
- Warmup (wash out initial transients) and Run (collect states)
- Output selection via `output_fraction` (10% of vertices for Ridge, 100% for HCNN)
- Ridge readout training and evaluation (R², NRMSE)
- HCNN readout training with cosine LR schedule
- Side-by-side comparison of both readouts on the same reservoir

**Expected output (abbreviated):**
```
=== HypercubeRC: Sine Wave Prediction ===

--- Ridge readout ---
  Config: N=128  Outputs=15 (10%)  Features=15
  R2:    ~1.000000   (effectively perfect)
  NRMSE: ~0.000xxx   (sub-0.1% error)

--- HCNN readout ---
  Config: N=128  raw state (all vertices)
  R2:    ~1.000000   (effectively perfect)
  NRMSE: ~0.00xxxx
```

**Make it yours:** Replace the sine wave generation with your own time series data.
Keep values in [-1, 1] (the reservoir clamps out-of-range inputs silently). Adjust
`DIM` to control reservoir size, and `warmup`/`collect` to match your data volume.

## SignalClassification

Classify four waveform types — sine, square, triangle, chirp — from reservoir state
alone. Two classifiers run side-by-side: Ridge (4 one-vs-rest readouts with argmax)
and HCNN (single 4-class softmax readout). Both produce a confusion matrix and
transition dynamics analysis showing how quickly the reservoir locks on after a
waveform switch.

**What it shows:**
- Reservoir as a feature extractor for pattern recognition
- Ridge: one-vs-rest classification with argmax over readout scores
- HCNN: native multi-class classification (softmax + cross-entropy)
- Confusion matrix and per-class accuracy breakdown
- Transition dynamics: accuracy vs steps after waveform switch

**Expected output (abbreviated):**
```
=== HypercubeRC: Signal Classification ===

Ridge config: DIM=7  N=128  Outputs=128 (70%)  Features=128

--- Ridge results ---
Overall accuracy: ~97%

--- HCNN results ---
Overall accuracy: ~97%

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

Config: DIM=8  N=256  Leak=0.3  Threshold=5x baseline
  Ridge: Outputs=128 (50%)  Features=128
  HCNN : Outputs=256 (100%)  raw state

--- Phase 1: Learn what "normal" looks like ---
  Baseline (prime test, RMSE):
    Ridge: ~0.0068   HCNN: ~0.0064

--- Phase 2: Monitor (30 windows of 200 steps) ---
  Window | Condition   |  Ridge RMSE / Ratio |  HCNN RMSE / Ratio | Status
      1  | Normal      |  ~0.007 /  ~1.0     |  ~0.006 / ~1.0     |
      6  | Noise spike |  ~0.080 / ~12.0     |  ~0.075 / ~12.0    | ** R+H **
     14  | DC drift    |  ~0.50  / ~75.0     |  ~0.40  / ~62.0    | ** R+H **
     22  | Freq shift  |  ~0.16  / ~24.0     |  ~0.13  / ~20.0    | ** R+H **
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
