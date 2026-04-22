# Examples

## BasicPrediction

The minimal hello-world for HypercubeRC. Demonstrates the complete pipeline on a
sine wave: drive the reservoir, collect states, train the HCNN readout, and evaluate.

This is the place to start if you want to understand how the pieces fit together.

**What it shows:**
- ESN construction with a single DIM template parameter
- Warmup (wash out initial transients) and Run (collect states)
- HCNN readout training with cosine LR schedule
- R² and NRMSE evaluation on held-out test set

**Expected output (abbreviated):**
```
=== HypercubeRC: Sine Wave Prediction ===

  Config: N=128  raw state (all vertices)
  R2:    ~1.000000   (effectively perfect)
  NRMSE: ~0.00xxxx   (sub-0.1% error)
```

**Make it yours:** Replace the sine wave generation with your own time series data.
Keep values in [-1, 1] (the reservoir clamps out-of-range inputs silently). Adjust
`DIM` to control reservoir size, and `warmup`/`collect` to match your data volume.

## SignalClassification

Classify four waveform types — sine, square, triangle, chirp — from reservoir state
alone. The HCNN readout performs native 4-class classification (softmax + cross-entropy),
producing a confusion matrix and transition dynamics analysis showing how quickly the
reservoir locks on after a waveform switch.

**What it shows:**
- Reservoir as a feature extractor for pattern recognition
- HCNN native multi-class classification
- Confusion matrix and per-class accuracy breakdown
- Transition dynamics: accuracy vs steps after waveform switch

**Expected output (abbreviated):**
```
=== HypercubeRC: Signal Classification ===

Overall accuracy: ~99%

  Steps after switch  | Accuracy
  0 - 3               |  ~96%
  Entire block        |  ~99%
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

Baseline (prime test, RMSE): ~0.0064   threshold ~0.032

  Window | Condition   |    RMSE     Ratio | Status
      1  | Normal      |  ~0.006     ~1.0  |
      6  | Noise spike |  ~0.075    ~12.0  | ** ANOMALY **
     14  | DC drift    |  ~0.40     ~62.0  | ** ANOMALY **
     22  | Freq shift  |  ~0.13     ~20.0  | ** ANOMALY **
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
