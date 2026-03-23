# Examples

## BasicPrediction

The minimal hello-world for HypercubeRC. Demonstrates the complete pipeline on a
sine wave: create a reservoir, drive it with input, apply the translation layer, train
a linear readout, and evaluate prediction quality.

This is the place to start if you want to understand how the pieces fit together.
The entire pipeline is visible in ~80 lines of straightforward C++.

**What it shows:**
- ESN construction with a single DIM template parameter
- Warmup (wash out initial transients) and Run (collect states)
- TranslationTransform to expand N states into 2.5N features
- LinearReadout training and evaluation (R2, NRMSE)
- Sample-by-sample prediction output

**Expected output (abbreviated):**
```
=== HypercubeRC: Basic Sine Wave Prediction ===

DIM=7  N=128  Features=320
Warmup=200  Collect=2000  Horizon=1

--- Results (test set: 600 samples) ---
  R2:    0.999999
  NRMSE: 0.001044

--- Sample predictions ---
  Step  |  Actual  | Predicted |   Error
   1400 |  +0.1209 |   +0.1208 |  +0.0002
   ...
```

**Make it yours:** Replace the sine wave generation (lines 50-53) with your own time
series data. Keep values in [-1, 1] (the reservoir clamps out-of-range inputs silently).
Adjust `DIM` to control reservoir size, and `warmup`/`collect` to match your data volume.

## StreamingAnomaly

Simulates industrial process monitoring with five phases: prime on normal data, monitor
for anomalies, detect degradation, adapt incrementally, and verify the adapted model.

This is the example to study if you're interested in real-time applications like
equipment monitoring, sensor drift detection, or process control.

**What it shows:**
- Batch training on historical "normal" data (priming the pump)
- Anomaly detection via prediction error exceeding a threshold (5x baseline RMSE)
- Side-by-side comparison of a frozen model vs an adapted model
- TrainIncremental for gradual drift tracking (blend=0.15)
- Why adaptation can't recover to baseline when noise increases (irreducible error floor)

**Expected output (abbreviated):**
```
=== HypercubeRC: Streaming Anomaly Detection ===

Signal: 0.6*sin(t) + 0.2*sin(3t) + noise + drift
Anomaly threshold: 5x baseline RMSE

--- Phase 1: Prime on normal operation ---
  Baseline RMSE: 0.007629

--- Phase 2: Streaming monitor (20 windows of 200 steps) ---

  Window | Condition  |  Frozen RMSE | Adapted RMSE | Status
      1  | Normal     |     0.008050 |     0.008050 |
     ...
     11  | Degrading  |     0.041782 |     0.040867 | ** ANOMALY DETECTED **
     ...
     20  | Degraded   |     0.075363 |     0.061543 | ANOMALY
```

**Make it yours:** Replace `GenerateProcess()` with your real sensor data feed.
Adjust `normal_noise` and `degraded_noise` to match your signal characteristics.
Tune the `anomaly_threshold` multiplier (5x is conservative; 3x catches subtler
changes). Adjust `blend` in `TrainIncremental` to control adaptation speed.

## Building

Both examples build automatically alongside the main benchmark:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/BasicPrediction
./build/StreamingAnomaly
```

In CLion, select the target from the run configuration dropdown (top toolbar).
