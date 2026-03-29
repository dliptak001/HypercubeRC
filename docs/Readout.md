# Readout

## Role in the Pipeline

    Reservoir (N states) -> TranslationLayer (2.5N features) -> **Readout**

The readout is the only trained component in the pipeline. The reservoir's random weights
are fixed at initialization; the translation layer is a fixed algebraic transform. All
learning happens in the readout, which fits a linear mapping from features to targets.

This is the core principle of reservoir computing: a complex, nonlinear dynamical system
(the reservoir) projects inputs into a high-dimensional space, and a simple linear model
(the readout) learns to extract the desired output from that projection. The reservoir
does the hard work of creating a rich representation; the readout does the easy work of
reading it.

## Two Implementations

HypercubeRC provides two readout classes. Both accept the same input format (row-major
float features + float targets) and expose the same prediction interface (`PredictRaw`,
`Predict`, `R2`, `Accuracy`). Both standardize features internally.

### LinearReadout

**Algorithm:** LMS/SGD (Least Mean Squares / Stochastic Gradient Descent) with L2 weight
decay and MSE-based pocket selection.

**How it works:**
1. Standardize features (zero mean, unit variance) using training set statistics.
2. For each epoch, iterate over training samples in shuffled order.
3. For each sample, compute the prediction error and update weights via the Widrow-Hoff
   rule: `w += lr * error * x - lr * weight_decay * w`.
4. After each epoch, compute MSE on the training set. If it's the lowest seen, save the
   current weights as the "pocket" (best-so-far).
5. Return the pocket weights after all epochs.

**Key parameters:**
- `learning_rate`: Step size. Default 0 = auto (1/num_features), which scales naturally
  with feature count.
- `max_epochs`: Default 200. More epochs rarely help due to pocket selection.
- `weight_decay`: Default 1e-4. Acts as online L2 regularization (equivalent to Ridge
  with time-varying lambda).
- `lr_decay`: Default 0.01. Reduces learning rate over epochs for convergence.

**Complexity:** O(num_features) per sample, O(num_features * num_samples * epochs) total.
No matrix operations — memory usage is O(num_features) for the weight vector.

**When to use:** DIM < 8 (fewer than ~640 translated features). At small feature counts,
SGD's implicit regularization via weight decay and early stopping matches or beats Ridge's
closed-form solution.

### RidgeRegression

**Algorithm:** Closed-form optimal linear solution via the normal equation with Tikhonov
regularization.

**How it works:**
1. Standardize features (zero mean, unit variance) using training set statistics.
2. Augment the feature matrix with a column of 1.0s for the bias term.
3. Compute the Gram matrix: `G = X'X + lambda * I` (bias column is NOT regularized).
4. Compute the cross-correlation: `c = X'y`.
5. Solve `G * w = c` via Gaussian elimination with partial pivoting (double precision).
6. The result is the globally optimal linear mapping for the given lambda.

**Key parameters:**
- `lambda`: Regularization strength. Default 1.0. Penalizes large weights to prevent
  overfitting, especially important when num_features approaches num_samples.

**Complexity:** O(num_features² * num_samples) for G computation, O(num_features³) for
the solve. Memory usage is O(num_features²) for the Gram matrix.

The Gram matrix computation is OpenMP-parallelized: each thread computes its own row of
the upper triangle, which is then mirrored. No write contention.

**When to use:** DIM >= 8 (640+ translated features). With enough features, the
closed-form global optimum outperforms SGD's approximate solution.

## Feature Standardization

Both readouts standardize features before training: each feature is shifted to zero mean
and scaled to unit variance. The statistics are computed from the training set and stored
internally; `PredictRaw` applies the same transform at inference time.

This is critical because the translation layer produces three feature classes with
different scale distributions:
- **x:** tanh output, approximately uniform in [-1, 1]
- **x²:** squared tanh, biased positive in [0, 1], smaller magnitude
- **x*x':** products, approximately uniform in [-1, 1]

Without standardization, Ridge regression's lambda penalizes the smaller-magnitude x²
features less than x features, leading to biased weight allocation. LinearReadout's
gradient updates would be dominated by the larger-scale features.

## Selection Policy

| DIM   | N      | Features (2.5N) | Recommended Readout | Rationale |
|-------|--------|-----------------|---------------------|-----------|
| 5     | 32     | 80              | LinearReadout       | SGD regularizes better at small sizes |
| 6     | 64     | 160             | LinearReadout       | Same |
| 7     | 128    | 320             | LinearReadout       | Crossover zone — both comparable |
| 8+    | 256+   | 640+            | RidgeRegression     | Closed-form optimum dominates |

In practice, all current diagnostics use LinearReadout across DIM 5-10 for consistency.
RidgeRegression is available for cases where optimal performance at DIM >= 8 matters
more than cross-DIM comparability.

## Streaming Mode (LinearReadout only)

For applications where data arrives continuously (e.g., process monitoring, anomaly
detection), LinearReadout supports incremental weight updates via `TrainIncremental()`:

```cpp
// Initial batch training on historical data
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

Internally, `TrainIncremental` trains a fresh model on the new data, then blends all
model parameters — weights, bias, feature mean, and feature scale — using exponential
moving average:

    W_updated = (1 - blend) * W_existing + blend * W_new

Blending the standardization statistics (mean, scale) alongside the weights allows the
model to track distribution drift in the input features over time.

RidgeRegression does not support streaming mode — its O(N³) solve cost makes it
impractical for incremental updates.

### Streaming Workflow: Priming the Pump

A streaming pipeline has three phases: prime, monitor, and adapt.

**Phase 1: Prime (batch training on historical data)**

Collect a representative batch of "normal" operation data — enough for the readout to
learn the baseline relationship. This is a standard batch `Train()` call. The batch
should be large enough for a good fit: at least 18*N samples (the project standard),
ideally more if the signal is noisy.

```cpp
// Drive reservoir through historical data (translation-optimized defaults)
float inp = Reservoir<8>::TranslationInputScaling();
ESN<8> esn(seed, ReadoutType::Linear, 1.0f,
           Reservoir<8>::TranslationSpectralRadius(), &inp);
esn.Warmup(historical_inputs, warmup_steps);
esn.Run(historical_inputs + warmup_steps, prime_steps);

// Translate and train readout
auto features = TranslationTransform<8>(esn.States(), prime_steps);
LinearReadout readout;
readout.Train(features.data(), targets, prime_steps, nf);
```

After priming, the readout has a baseline model of normal behavior. Use `PredictRaw()`
to verify it tracks the target signal with acceptable error.

**Phase 2: Monitor (predict and detect)**

Continue driving the reservoir with live data. For each new timestep (or small batch),
predict the expected output and compare to reality. The prediction error is the anomaly
signal — a spike indicates the process has deviated from the learned baseline.

```cpp
// Live loop
while (streaming) {
    esn.InjectInput(live_input);
    esn.Step();
    auto features = TranslationTransform<8>(esn.Outputs(), 1);
    float predicted = readout.PredictRaw(features.data());
    float error = actual_output - predicted;
    // error is the anomaly signal
}
```

No weight updates happen during monitoring. The model is frozen and acts as a reference.

**Phase 3: Adapt (incremental updates)**

When conditions drift gradually (e.g., electrode aging, seasonal changes), periodically
collect a new batch of recent data and blend it into the model. This allows the baseline
to track slow drift without losing the historical foundation.

```cpp
// Collect a window of recent data
esn.Run(recent_inputs, window_steps);
auto new_features = TranslationTransform<8>(esn.States(), window_steps);

// Blend into existing model — 10% new, 90% existing
readout.TrainIncremental(new_features.data(), recent_targets, window_steps, nf, 0.1f);
```

**How often to adapt** depends on the application:
- Slow drift (electrode aging): every N hours/days, blend=0.05-0.1
- Moderate drift (seasonal): every N minutes, blend=0.1-0.3
- Fast regime change: single update with blend=0.5-1.0

**Key principle:** The prime batch should be large and representative. Incremental
updates should be smaller batches at low blend values. If the process changes
fundamentally (not drift, but a new regime), re-prime with a fresh `Train()` call.

For a complete working implementation of this workflow, see
[examples/StreamingAnomaly.cpp](../examples/StreamingAnomaly.cpp).

## Common Interface

Both readouts expose:

| Method | Returns | Description |
|--------|---------|-------------|
| `Train(features, targets, n, nf)` | void | Fit weights from training data (batch) |
| `TrainIncremental(features, targets, n, nf, blend)` | void | Incremental update (LinearReadout only) |
| `PredictRaw(features)` | float | Continuous prediction (standardizes internally) |
| `Predict(features)` | float | Thresholded to {-1.0, +1.0} |
| `R2(features, targets, n)` | double | Coefficient of determination on test set |
| `Accuracy(features, labels, n)` | double | Classification accuracy on test set |
| `Weights()` | vector | Learned weight vector |
| `Bias()` | float/double | Learned bias term |

## Implementation Notes

- Both classes are in `readout/` with separate .h/.cpp files.
- LinearReadout stores weights as `float`; RidgeRegression uses `double` internally for
  numerical stability in the Gram matrix solve.
- Neither class is templated — they accept arbitrary feature counts at runtime.
- Neither class stores the training data — only the learned weights and standardization
  statistics.
