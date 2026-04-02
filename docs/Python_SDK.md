# HypercubeRC Python SDK

Python bindings for reservoir computing on Boolean hypercube graphs.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [The dim parameter](#the-dim-parameter)
  - [Enums](#enums)
  - [ESN](#esn)
- [Input Data Layout](#input-data-layout)
- [Dependencies](#dependencies)

## Installation

Requirements: Python 3.9+, a C++23 compiler with OpenMP support (GCC 13+, Clang 17+), CMake 3.20+, Ninja.

### Linux / macOS

```bash
cd python
pip install .
```

### Windows (MinGW)

The build requires GCC (MinGW) — not MSVC — because the project uses OpenMP features that MSVC does not support. Set the compiler and generator environment variables before building:

```powershell
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install scikit-build-core pybind11 numpy cmake
cd python
pip install . --no-build-isolation
```

The resulting `.pyd` statically links all MinGW runtime libraries (libgomp, libstdc++, libgcc) so no MinGW DLLs are needed at runtime.

### Development install

```bash
pip install -e .
```

### Running tests

```bash
pip install ".[test]"
pytest python/tests/
```

## Quick Start

```python
import numpy as np
import hypercube_rc as hrc

# Generate a sine wave
signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)

# Create an ESN with 2^7 = 128 neurons
esn = hrc.ESN(dim=7, seed=42)

# Drive the reservoir
esn.warmup(signal[:200])          # wash out initial transient
esn.run(signal[200:-1])           # record states

# Train on next-step prediction
targets = signal[201:]
train_size = 1400
test_size = esn.num_collected - train_size

esn.train(targets[:train_size])

# Evaluate
r2 = esn.r2(targets, train_size, test_size)
nrmse = esn.nrmse(targets, train_size, test_size)
print(f"R² = {r2:.6f}, NRMSE = {nrmse:.6f}")  # ~1.0000, ~0.001
```

---

## API Reference

### The `dim` parameter

`dim` controls the hypercube dimension. The reservoir has N = 2^dim neurons. Supported values: 5-12.

| dim | Neurons | Typical use |
|-----|---------|-------------|
| 5   | 32      | Fast prototyping, embedded |
| 6   | 64      | Light benchmarks |
| 7   | 128     | Standard benchmarks |
| 8   | 256     | Production, complex tasks |
| 9-12 | 512-4096 | Research, high-capacity tasks |

For dim 9+, reduce `output_fraction` to control Ridge readout cost (e.g., 0.25 for dim 10).

---

### Enums

#### `ReadoutType`

| Value | Description |
|-------|-------------|
| `ReadoutType.Ridge` | Closed-form Ridge regression. Deterministic, fast, optimal for the given regularization. Default. |
| `ReadoutType.Linear` | Online SGD with L2 decay and pocket selection. Supports streaming via `train_incremental()`. |

#### `FeatureMode`

| Value | Description |
|-------|-------------|
| `FeatureMode.Translated` | Expands M selected states into 2.5M features via [x \| x² \| x·x_antipodal]. Reduces NRMSE by 20-70% on standard benchmarks. Default. |
| `FeatureMode.Raw` | Uses M selected states directly. Fewer features, faster computation, sufficient for simple tasks. |

---

### ESN

The complete API. Owns the full Reservoir → Translation → Readout pipeline.

```python
import hypercube_rc as hrc

# Construction
esn = hrc.ESN(dim=7)                                                    # defaults
esn = hrc.ESN(dim=7, readout_type=hrc.ReadoutType.Linear,
              feature_mode=hrc.FeatureMode.Raw)                          # explicit

# Reservoir driving
esn.warmup(inputs)                # drive without recording
esn.run(inputs)                   # drive and collect states
esn.clear_states()                # clear collected data (keeps readout)

# Training
esn.train(targets)                # default parameters
esn.train(targets, lambda_=0.1)   # Ridge: custom regularization
esn.train(targets, lr=0.01, epochs=300)  # Linear: custom SGD
esn.train_incremental(targets, blend=0.1)  # Linear: streaming update

# Prediction & evaluation
esn.predict_raw(timestep)         # single continuous prediction
esn.predictions()                 # all predictions as ndarray
esn.r2(targets, start, count)     # R-squared
esn.nrmse(targets, start, count)  # normalized RMSE
esn.accuracy(labels, start, count)  # classification accuracy

# State access
esn.selected_states()             # stride-selected states as ndarray
```

---

#### Construction

```python
ESN(dim, *, seed=0, spectral_radius=0.9, input_scaling=0.02,
    leak_rate=1.0, alpha=1.0, num_inputs=1, output_fraction=1.0,
    readout_type=ReadoutType.Ridge, feature_mode=FeatureMode.Translated)
```

Creates the reservoir, initializes the selected readout type, and computes output selection parameters from `output_fraction`. The reservoir weights are generated and spectral-radius-rescaled at construction time.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | — | Hypercube dimension (5-12). N = 2^dim neurons. |
| `seed` | `int` | `0` | RNG seed for weight initialization. Every seed produces a valid topology. |
| `spectral_radius` | `float` | `0.9` | Target spectral radius. Scale-invariant across all dim values. |
| `input_scaling` | `float` | `0.02` | Input weight magnitude, U(-input_scaling, +input_scaling). Scale-invariant. |
| `leak_rate` | `float` | `1.0` | Leaky integrator coefficient. 1.0 = full replacement. < 1.0 adds smoothing. |
| `alpha` | `float` | `1.0` | Gain inside tanh: `tanh(alpha * sum)`. > 1.0 sharpens nonlinearity. |
| `num_inputs` | `int` | `1` | Number of input channels. Channel k drives every K-th vertex starting at offset k. |
| `output_fraction` | `float` | `1.0` | Fraction of N vertices used as readout features, in (0.0, 1.0]. |
| `readout_type` | `ReadoutType` | `Ridge` | Which readout to use. |
| `feature_mode` | `FeatureMode` | `Translated` | Whether to apply the translation layer. |

---

#### Reservoir Driving

##### `warmup(inputs)`

Drive the reservoir for a number of timesteps without recording state. Use this to wash out the reservoir's initial transient (zero state) before collecting data for training.

**Parameters:**
- `inputs` — NumPy array of input values. Shape `(num_steps,)` for single-input or `(num_steps, num_inputs)` for multi-input. Converted to float32 automatically.

**Notes:**
- Does not allocate memory or record states.
- The reservoir's internal state is updated in-place.

---

##### `run(inputs)`

Drive the reservoir and record the full N-dimensional state vector at each step. States are appended — multiple `run()` calls accumulate.

**Parameters:**
- `inputs` — NumPy array. Same shape convention as `warmup()`.

**Notes:**
- After `run()`, collected states are available for training and evaluation.
- Features are computed lazily when first needed (by `train()`, `r2()`, etc.).

---

##### `clear_states()`

Clear all collected states and cached features. The reservoir's live internal state is **not** reset — it retains its current activation. The trained readout is also preserved.

Use this between independent sequences: clear the collected data, then `warmup()` + `run()` on a new input sequence without rebuilding the ESN.

---

#### Training

##### `train(targets, *, lambda_=None, lr=None, epochs=None, weight_decay=1e-4, lr_decay=0.01)`

Train the readout on the first `len(targets)` collected states.

**Dispatch rules:**
- No optional args → default parameters for the selected readout type
- `lambda_` → Ridge with custom regularization (asserts Ridge readout)
- `lr` → Linear SGD with custom parameters (asserts Linear readout)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `targets` | `ndarray` | — | Target values, shape `(train_size,)`. Regression: continuous. Classification: {-1, +1}. |
| `lambda_` | `float` | `None` | Ridge regularization strength. Typical range: 0.01-100. |
| `lr` | `float` | `None` | Learning rate. 0.0 = auto (1/num_features). |
| `epochs` | `int` | `None` | Number of SGD epochs. Default: 200 when using Linear. |
| `weight_decay` | `float` | `1e-4` | L2 regularization for SGD. |
| `lr_decay` | `float` | `0.01` | LR decay factor. Effective LR at epoch e = lr / (1 + lr_decay × e). |

**Notes:**
- Triggers feature computation if not already done.
- For Ridge, calling `train()` again replaces the previous solution entirely.
- For Linear, calling `train()` again retrains from scratch (use `train_incremental()` for streaming).

---

##### `train_incremental(targets, *, blend=0.1, lr=0.0, epochs=200, weight_decay=1e-4, lr_decay=0.01)`

Incrementally update the Linear readout for streaming applications. Asserts Linear readout.

Trains a fresh model on the provided data, then blends it with the existing model:

```
W_updated = (1 - blend) * W_existing + blend * W_new
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `targets` | `ndarray` | — | Target values for the new data window. |
| `blend` | `float` | `0.1` | Blending factor in (0, 1]. At 1.0, fully replaces old model. |
| `lr` | `float` | `0.0` | Learning rate. 0.0 = auto. |
| `epochs` | `int` | `200` | SGD epochs for the fresh model. |
| `weight_decay` | `float` | `1e-4` | L2 regularization. |
| `lr_decay` | `float` | `0.01` | LR decay factor. |

**Notes:**
- If no prior `train()` has been called, delegates to `train()` (blend is ignored).
- Typical: call `run()` with a new data window, then `train_incremental()` to adapt.

---

#### Prediction and Evaluation

##### `predict_raw(timestep) → float`

Return the raw continuous prediction for a single collected timestep.

**Parameters:**
- `timestep` — Index into collected states, in [0, num_collected).

**Returns:** Continuous float prediction. For classification, threshold at 0.0.

---

##### `predictions() → ndarray`

Return predictions for all collected timesteps as a 1D float32 array of shape `(num_collected,)`.

---

##### `r2(targets, start, count) → float`

Compute R-squared (coefficient of determination) on a slice of collected states.

```
R² = 1 - SS_res / SS_tot
```

**Parameters:**
- `targets` — Target array. The slice `targets[start:start+count]` is used.
- `start` — First timestep index.
- `count` — Number of timesteps to evaluate.

**Returns:** R² value. 1.0 = perfect. 0.0 = predicts the mean. Can be negative.

**Typical usage (train/test split):**

```python
esn.train(targets[:train_size])
test_r2 = esn.r2(targets, train_size, test_size)
```

---

##### `nrmse(targets, start, count) → float`

Compute Normalized Root Mean Squared Error on a slice of collected states.

```
NRMSE = sqrt(MSE) / sqrt(Var(targets))
```

**Parameters:**
- `targets` — Target array. Slice `targets[start:start+count]` is used.
- `start` — First timestep index.
- `count` — Number of timesteps to evaluate.

**Returns:** NRMSE value. 0.0 = perfect. 1.0 = as bad as predicting the mean.

---

##### `accuracy(labels, start, count) → float`

Compute classification accuracy on a slice of collected states. Predictions are thresholded at 0.0.

**Parameters:**
- `labels` — Label array with values {-1.0, +1.0}.
- `start` — First timestep index.
- `count` — Number of timesteps to evaluate.

**Returns:** Fraction correct in [0.0, 1.0].

---

#### State and Feature Access

##### `selected_states() → ndarray`

Extract stride-selected vertices from all collected states.

**Returns:** Array of shape `(num_collected, num_output_verts)`, dtype float32.

---

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dim` | `int` | Hypercube dimension. |
| `N` | `int` | Number of neurons (2^dim). |
| `num_collected` | `int` | Timesteps recorded by `run()`. |
| `num_features` | `int` | Features per timestep. M for Raw, 2.5M for Translated. |
| `num_inputs` | `int` | Number of input channels. |
| `output_fraction` | `float` | Fraction of vertices used as readout features. |
| `output_stride` | `int` | Stride used for vertex selection: max(1, N/M). |
| `num_output_verts` | `int` | Number of selected output vertices M. |
| `readout_type` | `ReadoutType` | Readout type selected at construction. |
| `feature_mode` | `FeatureMode` | Feature mode selected at construction. |
| `alpha` | `float` | Tanh gain parameter. |

---

## Input Data Layout

Input arrays follow row-major layout:

**Single-input** (num_inputs=1):
```python
inputs = signal[200:400]  # shape (200,) — 200 timesteps
```

**Multi-input** (num_inputs=K):
```python
inputs = np.column_stack([ch1, ch2, ch3])  # shape (num_steps, 3)
```

Each row contains one value per channel. The array is flattened internally to match the C++ convention: `[step0_ch0, step0_ch1, ..., step1_ch0, ...]`.

Arrays of any numeric dtype are automatically converted to C-contiguous float32.

---

## Dependencies

**Runtime:** NumPy >= 1.21

**Build time:** scikit-build-core >= 0.10, pybind11 >= 2.13, C++23 compiler with OpenMP, CMake 3.20+

The library uses OpenMP internally for parallelization. No other external dependencies.
