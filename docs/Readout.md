# Readout — The Only Trained Component

## Role in the pipeline

```
Reservoir (N states) ──> Translation Layer (2.5N features) ──> Readout
    fixed random              fixed algebra                    TRAINED
```

The readout is the only trained component in the entire pipeline. The
reservoir's random weights are fixed at initialization; the translation
layer is a fixed algebraic transform. All learning happens here — the
readout fits a linear mapping from features to targets.

This is the core principle of reservoir computing: a complex, nonlinear
dynamical system (the reservoir) projects inputs into a high-dimensional
space, and a simple linear model (the readout) learns to extract the
desired output from that projection. The reservoir does the hard work
of creating a rich representation; the readout does the easy work of
reading it.

## Two implementations

HypercubeRC provides two readout classes. Both accept the same input
format (row-major float features + float targets), expose the same
prediction interface (`PredictRaw`, `Predict`, `R2`, `Accuracy`), and
standardize features internally. The choice between them is a
speed-vs-optimality tradeoff.

### LinearReadout — online SGD

**Algorithm:** LMS/SGD (Least Mean Squares / Stochastic Gradient Descent)
with L2 weight decay and MSE-based pocket selection.

**How it works:**
1. Standardize features (zero mean, unit variance) from training set.
2. For each epoch, iterate over training samples in shuffled order.
3. For each sample, compute prediction error and update weights:
   `w += lr * error * x - lr * weight_decay * w`
4. After each epoch, check MSE on the training set. If it's the lowest
   seen, save the current weights as the "pocket" (best-so-far).
5. Return the pocket weights after all epochs.

**Key parameters:**

| Parameter | Default | Role |
|-----------|---------|------|
| `learning_rate` | 0 (auto: 1/num_features) | Step size, scales with feature count |
| `max_epochs` | 200 | Training iterations; pocket selection means more rarely helps |
| `weight_decay` | 1e-4 | Online L2 regularization (shrinks weights toward zero) |
| `lr_decay` | 0.01 | Reduces learning rate over epochs for convergence |

**Cost:** O(num_features * num_samples * epochs). No matrix operations —
memory is O(num_features) for the weight vector.

**When to use:** DIM < 8 (fewer than ~640 translated features). At small
feature counts, SGD's implicit regularization matches or beats Ridge.
Also the only option for streaming/incremental updates.

### RidgeRegression — closed-form optimum

**Algorithm:** Normal equations with Tikhonov regularization, solved via
Gaussian elimination with partial pivoting in double precision.

**How it works:**
1. Standardize features (zero mean, unit variance) from training set.
2. Augment the feature matrix with a bias column of 1.0s.
3. Build the Gram matrix: `G = X'X + lambda * I` (bias NOT regularized).
4. Build the cross-correlation: `c = X'y`.
5. Solve `G * w = c` via Gaussian elimination with partial pivoting.
6. The result is the globally optimal linear mapping for the given lambda.

**Key parameters:**

| Parameter | Default | Role |
|-----------|---------|------|
| `lambda` | 1.0 | Regularization strength; prevents overfitting when features ~ samples |

**Cost:** O(num_features² * num_samples) for G, O(num_features³) for
the solve. Memory is O(num_features²) for the Gram matrix. The G
computation is OpenMP-parallelized (each thread owns a row of the upper
triangle, mirrored afterward — no write contention).

**When to use:** DIM >= 8 (640+ translated features). The closed-form
global optimum consistently outperforms SGD at larger feature counts.
This is the default for the main benchmarks (MG, NARMA-10).

## Feature standardization

Both readouts standardize features before training: each feature is
shifted to zero mean and scaled to unit variance. The statistics are
computed from the training set and stored; `PredictRaw` applies the
same transform at inference.

This is critical because the translation layer produces three feature
classes with different scales:

| Class | Range | Without standardization... |
|-------|-------|--------------------------|
| x     | [-1, +1] | Dominates gradient/regularization (largest magnitude) |
| x²    | [0, 1] | Under-penalized by Ridge lambda (smaller magnitude) |
| x*x'  | [-1, +1] | Comparable to x |

Without standardization, Ridge's lambda penalizes feature groups
unevenly, and SGD's gradient is dominated by the largest-scale features.

## Which readout should I use?

| DIM | N    | Features (2.5N) | Recommended | Rationale |
|-----|------|-----------------|-------------|-----------|
| 5   | 32   | 80              | LinearReadout | SGD regularizes well at small sizes |
| 6   | 64   | 160             | LinearReadout | Same |
| 7   | 128  | 320             | Either | Crossover zone — both comparable |
| 8+  | 256+ | 640+            | RidgeRegression | Closed-form optimum dominates |

The main benchmark suite uses Ridge for MG and NARMA-10 (where optimal
accuracy matters) and Linear for Memory Capacity (the standard metric).
Diagnostics default to Linear but accept a `ReadoutType` parameter.

## Streaming mode (LinearReadout only)

For applications where data arrives continuously (e.g., process
monitoring, anomaly detection), LinearReadout supports incremental
weight updates via `TrainIncremental()`:

```cpp
// Initial batch training
readout.Train(features_batch1, targets_batch1, n1, nf);

// New data arrives — blend into existing model
readout.TrainIncremental(features_batch2, targets_batch2, n2, nf, 0.1f);
```

The `blend` parameter controls how much the new data influences the model:

| blend | Behavior | Use case |
|-------|----------|----------|
| 1.0   | Full replacement (same as Train) | Reset to new data |
| 0.5   | Equal mix of old and new | Regime change |
| 0.1   | 10% new, 90% existing | Gradual adaptation |
| 0.01  | 1% new, 99% existing | Slow drift tracking |

Internally, `TrainIncremental` trains a fresh model on the new data,
then blends all model parameters — weights, bias, feature mean, and
feature scale:

```
W_updated = (1 - blend) * W_existing + blend * W_new
```

Blending the standardization statistics alongside the weights allows
the model to track distribution drift in the input features over time.

**Note:** `TrainIncremental` requires the same feature count as the
original `Train` call. A mismatch throws `std::invalid_argument`.

RidgeRegression does not support streaming — its O(N³) solve cost
makes incremental updates impractical.

### Streaming workflow: prime, monitor, adapt

**Phase 1 — Prime** (batch training on historical data):

Collect a representative batch of "normal" operation data and train
the readout. This is a standard batch `Train()` call with at least
18*N samples (the project standard).

```cpp
auto cfg = ReservoirDefaults<8>::MakeConfig(seed, FeatureMode::Translation);
ESN<8> esn(cfg, ReadoutType::Ridge);
esn.Warmup(historical_data, 500);
esn.Run(historical_data + 500, prime_steps);

auto features = TranslationTransform<8>(esn.States(), prime_steps);
RidgeRegression readout;
readout.Train(features.data(), targets, prime_steps, nf);
```

**Phase 2 — Monitor** (predict and detect):

Drive the reservoir with live data in windows. For each window, predict
the expected output and compare to reality. The prediction error is the
anomaly signal.

```cpp
// Process a window of live data
esn.ClearStates();
esn.Run(live_window, window_size);
auto features = TranslationTransform<8>(esn.States(), window_size);

for (size_t t = 0; t < window_size; ++t) {
    float predicted = readout.PredictRaw(features.data() + t * nf);
    float error = actual[t] - predicted;
    // error is the anomaly signal
}
```

No weight updates during monitoring. The model is frozen.

**Phase 3 — Adapt** (incremental updates, LinearReadout only):

When conditions drift gradually, periodically collect recent data and
blend it into the model:

```cpp
LinearReadout lr;
lr.Train(initial_features, initial_targets, n_init, nf);

// Later, when drift is detected:
lr.TrainIncremental(recent_features, recent_targets, n_recent, nf, 0.1f);
```

**How often to adapt:**
- Slow drift (sensor aging): every N hours/days, blend=0.05-0.1
- Moderate drift (seasonal): every N minutes, blend=0.1-0.3
- Fast regime change: single update with blend=0.5-1.0, or re-prime

For a complete working implementation, see
`examples/StreamingAnomaly.cpp` and `examples/StreamingAnomaly.md`.

## Common interface

Both readouts expose:

| Method | Returns | Description |
|--------|---------|-------------|
| `Train(features, targets, n, nf)` | void | Fit weights from training data |
| `TrainIncremental(...)` | void | Incremental update (LinearReadout only) |
| `PredictRaw(features)` | float | Continuous prediction (standardized internally) |
| `Predict(features)` | float | Thresholded to {-1.0, +1.0} |
| `R2(features, targets, n)` | double | Coefficient of determination |
| `Accuracy(features, labels, n)` | double | Classification accuracy |
| `Weights()` | vector | Learned weight vector |
| `Bias()` | float/double | Learned bias term |
| `NumFeatures()` | size_t | Feature count this model was trained on |

## Implementation notes

- Both classes live in `readout/` with separate .h/.cpp files.
- LinearReadout stores weights as `float`; RidgeRegression uses `double`
  internally for numerical stability in the Gram matrix solve.
- Neither class is templated — they accept arbitrary feature counts at
  runtime.
- Neither class stores the training data — only the learned weights and
  standardization statistics.
