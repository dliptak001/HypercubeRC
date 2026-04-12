# Readout — The Only Trained Component

## Role in the pipeline

```
Linear / Ridge path:
Reservoir (N states) ──> Translation Layer (2.5N features) ──> Readout
    fixed random              fixed algebra                    TRAINED

HCNN path:
Reservoir (N states) ────────────────────────────────────────> CNNReadout
    fixed random                                                TRAINED
```

The readout is the only trained component in the entire pipeline. The
reservoir's random weights are fixed at initialization; the translation
layer is a fixed algebraic transform. All learning happens here.

Linear and Ridge readouts fit a linear mapping from translated features
to targets. The CNN readout replaces that linear fit with a learned
convolutional network that operates directly on raw reservoir state,
discovering its own nonlinear features instead of relying on the
hand-crafted translation expansion.

This is the core principle of reservoir computing: a complex, nonlinear
dynamical system (the reservoir) projects inputs into a high-dimensional
space, and a simple model (the readout) learns to extract the desired
output from that projection.

## Three implementations

HypercubeRC provides three readout classes:

| Class             | Features consumed | Training                  | Best for                         |
|-------------------|-------------------|---------------------------|----------------------------------|
| `LinearReadout`   | Translated (2.5M) | Online SGD + pocket       | Small DIM, streaming, incremental|
| `RidgeRegression` | Translated (2.5M) | Closed-form normal equations | Large DIM batch regression    |
| `CNNReadout`      | Raw state (N)     | Mini-batch backprop (Adam)| Hard tasks, classification, larger DIM |

All three expose the same prediction interface (`PredictRaw`, `R2`,
`Accuracy`, `Weights`, `Bias`) so ESN can dispatch uniformly, and all
three standardize their inputs internally. The differences lie in what
features they see and how the training signal turns into parameters.

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
computation fills the upper triangle, mirrored afterward.

**When to use:** DIM >= 8 (640+ translated features). The closed-form
global optimum consistently outperforms SGD at larger feature counts.
This is the default for the main benchmarks (MG, NARMA-10).

### CNNReadout — learned convolutional readout

**Algorithm:** HypercubeCNN — a stack of hypercube Conv+MaxPool layers
followed by global average pooling and a dense head, trained with Adam
and cosine-annealed learning rate. Operates directly on raw reservoir
state; the translation layer is bypassed entirely.

**How it works:**
1. Standardize the raw N-vertex state (per-vertex mean/std) from the
   training set.
2. Auto-size the architecture from DIM: `min(DIM - 3, 4)` Conv+Pool
   pairs, base `conv_channels` doubling per layer (16 → 32 → 64 → 128).
   Each Pool halves the hypercube dimension, so the stack depth is
   capped by `DIM - 3` (HCNNConv requires DIM ≥ 3).
3. For each epoch, shuffle samples into mini-batches and run Adam
   forward/backward over the full training set. Learning rate follows
   a cosine schedule from `lr_max` down to `lr_max * lr_min_frac`.
4. Supports two task heads:
   - **Regression** — per-output target centering, MSE loss, de-centered
     predictions at inference.
   - **Classification** — integer class labels, softmax+cross-entropy,
     logit output via `PredictRaw` or argmax via `PredictClass`.
5. After training, weights are flattened via `HCNN::GetWeights()` for
   serialization and restored via `SetWeights()` on reload.

**Key parameters (`CNNReadoutConfig`):**

| Parameter       | Default                   | Role |
|-----------------|---------------------------|------|
| `num_outputs`   | 1                         | Regression targets, or class count |
| `task`          | `HCNNTask::Regression`    | Regression or Classification |
| `num_layers`    | 0 (auto: `min(DIM-3, 4)`) | Conv+Pool pairs; manual override |
| `conv_channels` | 16                        | Base channel count; doubles per layer |
| `epochs`        | 200                       | Full passes over the training set |
| `batch_size`    | 32                        | Mini-batch size |
| `lr_max`        | 0.005                     | Peak learning rate (cosine schedule) |
| `lr_min_frac`   | 0.1                       | Floor as fraction of `lr_max` |
| `weight_decay`  | 0.0                       | L2 weight decay |
| `seed`          | 42                        | Weight initialization seed |
| `verbose`       | false                     | Print per-epoch training accuracy |

**Cost:** O(epochs * samples * layer_flops). For a typical DIM=8
configuration (~256 states per sample, 4 Conv+Pool pairs, ~20k samples,
300 epochs) this runs in seconds to minutes depending on core count.
CPU cores saturate at `batch_size >= 128`.

**Stability note:** `lr_max` above ~0.003 can drive weights into
denormal/NaN territory, where CPU falls off fast math paths and
throughput collapses. Benchmark defaults (`diagnostics/MackeyGlass.h`,
`diagnostics/NARMA10.h`) use `epochs=300, batch_size=128, lr_max=0.003`.

**When to use:**
- Tasks where the linear-readout ceiling is hit and nonlinear feature
  discovery is worth the training cost. NARMA-10 at DIM 7+ and the
  signal-classification example both show HCNN matching or beating
  Ridge with translation features.
- Classification problems. Ridge handles classification as one-vs-rest
  regression + argmax; HCNN natively supports multi-class with
  softmax+CE.
- DIM 7+ where the auto-sized architecture gets enough Conv+Pool depth
  to be expressive. At DIM 5-6 the single Conv+Pool pair typically
  underperforms Ridge.

**When not to use:**
- Streaming / incremental training. HCNN is batch only — use
  LinearReadout for online adaptation.
- Small-DIM tasks (5-6) where Ridge is cheaper and at least as good.

See `readout/CNNReadout.md` for the full architecture table and
integration-status breakdown, and `examples/BasicPrediction.cpp` /
`examples/SignalClassification.cpp` for working side-by-side code.

## Feature standardization

All three readouts standardize their inputs before training: each
feature (or raw vertex, for HCNN) is shifted to zero mean and scaled
to unit variance. The statistics are computed from the training set
and stored; `PredictRaw` applies the same transform at inference.

For Linear/Ridge this is critical because the translation layer
produces three feature classes with different scales:

| Class | Range | Without standardization... |
|-------|-------|--------------------------|
| x     | [-1, +1] | Dominates gradient/regularization (largest magnitude) |
| x²    | [0, 1] | Under-penalized by Ridge lambda (smaller magnitude) |
| x*x'  | [-1, +1] | Comparable to x |

Without standardization, Ridge's lambda penalizes feature groups
unevenly, and SGD's gradient is dominated by the largest-scale features.

For HCNN the raw N-vertex state is centered and scaled per-vertex
before it enters the first Conv layer. The translation layer is
bypassed — HCNN's convolution kernels discover their own nonlinear
features from raw state, which is the whole point of using it.

## Which readout should I use?

| DIM | N    | Linear features (2.5N) | Recommended defaults              |
|-----|------|------------------------|-----------------------------------|
| 5   | 32   | 80                     | LinearReadout (SGD regularizes well) |
| 6   | 64   | 160                    | LinearReadout / Ridge             |
| 7   | 128  | 320                    | Ridge or HCNN (HCNN starts being competitive) |
| 8+  | 256+ | 640+                   | Ridge for fast closed-form; HCNN when accuracy ceiling matters |

The main benchmark suite uses Ridge for MG and NARMA-10 (where optimal
accuracy matters) and Linear for Memory Capacity (the standard metric).
Adding `--hcnn` to the suite runs HCNN alongside Ridge for
apples-to-apples comparison.

For classification tasks, CNNReadout is the natural fit — a single
multi-class readout replaces N one-vs-rest Ridge heads and tends to
produce cleaner lock-on at block transitions. See
`examples/SignalClassification.cpp`.

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

CNNReadout also does not support streaming. Multi-epoch mini-batch
training is intrinsic to the algorithm; there is no cheap "blend new
data into existing weights" path that preserves optimization state.
Use LinearReadout for any application that needs incremental updates.

### Streaming workflow: prime, monitor, adapt

**Phase 1 — Prime** (batch training on historical data):

Collect a representative batch of "normal" operation data and train
the readout. This is a standard batch `Train()` call with at least
18*N samples (the project standard).

```cpp
ReservoirConfig cfg;
cfg.seed = seed;
ESN<8> esn(cfg, ReadoutType::Ridge, FeatureMode::Translated);
esn.Warmup(historical_data, 500);
esn.Run(historical_data + 500, prime_steps);
esn.Train(targets, train_size);
```

**Phase 2 — Monitor** (predict and detect):

Drive the reservoir with live data in windows. For each window, predict
the expected output and compare to reality. The prediction error is the
anomaly signal. `ClearStates()` resets collected states but preserves the
trained readout weights.

```cpp
// Process a window of live data
esn.ClearStates();
esn.Run(live_window, window_size);

for (size_t t = 0; t < window_size; ++t) {
    float predicted = esn.PredictRaw(t);
    float error = actual[t] - predicted;
    // error is the anomaly signal
}
```

No weight updates during monitoring. The model is frozen.

**Phase 3 — Adapt** (incremental updates, LinearReadout only):

When conditions drift gradually, periodically collect recent data and
blend it into the model:

```cpp
ESN<8> esn(cfg, ReadoutType::Linear, FeatureMode::Translated);
// ... warmup, run, initial train ...

// Later, when drift is detected:
esn.TrainIncremental(recent_targets, recent_size, /*blend=*/0.1f);
```

**How often to adapt:**
- Slow drift (sensor aging): every N hours/days, blend=0.05-0.1
- Moderate drift (seasonal): every N minutes, blend=0.1-0.3
- Fast regime change: single update with blend=0.5-1.0, or re-prime

For a complete working implementation, see
`examples/StreamingAnomaly.cpp` and `examples/StreamingAnomaly.md`.

## Common interface

All three readouts expose the subset of this interface that makes
sense for them (ESN's `std::visit` lambdas only call methods that
exist on every alternative):

| Method                              | Returns | Linear | Ridge | CNN |
|-------------------------------------|---------|--------|-------|-----|
| `Train(features, targets, n, nf)`   | void    | Y      | Y     | Y (via CNN-specific config) |
| `TrainIncremental(...)`             | void    | Y      | —     | —   |
| `PredictRaw(features)`              | float   | Y      | Y     | Y (scalar, single-output) |
| `PredictRaw(features, out)`         | void    | —      | —     | Y (multi-output) |
| `PredictClass(features)`            | int     | —      | —     | Y   |
| `R2(features, targets, n)`          | double  | Y      | Y     | Y   |
| `Accuracy(features, labels, n)`     | double  | Y      | Y     | Y   |
| `Weights()`                         | vector  | Y      | Y     | Y (flattened blob) |
| `Bias()`                            | float/double | Y | Y  | Y (target mean fallback) |
| `NumFeatures()`                     | size_t  | Y      | Y     | Y (= N for raw state) |
| `NumOutputs()`                      | size_t  | —      | —     | Y   |

ESN wraps the three readouts in a `std::variant` and dispatches via
`std::visit`; call sites see a uniform `ESN::Train / PredictRaw / R2`
interface regardless of which readout is selected. HCNN has a
dedicated `Train(targets, n, CNNReadoutConfig)` overload since it
needs its own hyperparameters.

## Implementation notes

- All three classes live in `readout/` with separate .h/.cpp files.
- LinearReadout stores weights as `float`; RidgeRegression uses `double`
  internally for numerical stability in the Gram matrix solve.
- CNNReadout holds a `std::unique_ptr<hcnn::HCNN>` via PIMPL so that
  `#include "HCNN.h"` stays in the .cpp only.
- None of the classes are templated — they accept arbitrary feature
  counts at runtime.
- None of the classes store the training data — only the learned
  weights, standardization statistics, and (for CNN) the config used
  to rebuild the network on reload.
