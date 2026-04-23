# HypercubeRC Python SDK

Python bindings for reservoir computing on Boolean hypercube graphs.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [The dim parameter](#the-dim-parameter)
  - [ESN](#esn)
  - [Streaming / Online Training](#streaming--online-training)
- [Input Data Layout](#input-data-layout)
- [Dependencies](#dependencies)

## Installation

### From PyPI (recommended)

Pre-built wheels are available for Python 3.10-3.13 on Windows (x64),
Linux (x86_64, aarch64), and macOS (x86_64, arm64):

```bash
pip install hypercube-rc
```

### From source

Requirements: Python 3.10+, a C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+), CMake 3.20+.

```bash
git clone https://github.com/dliptak001/HypercubeRC.git
cd HypercubeRC/python
pip install .
```

On Windows with MinGW, install build dependencies and set compiler environment
variables before building:

```powershell
pip install scikit-build-core pybind11 numpy
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install . --no-build-isolation
```

### Running tests

```bash
pip install ".[test]"
pytest python/tests/
```

## Quick Start

### Simple (recommended)

```python
import numpy as np
import hypercube_rc as hrc

signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)

esn = hrc.ESN(dim=7, seed=42)
esn.fit(signal, warmup=200)       # warmup, run, train in one call

print(f"R² = {esn.r2():.6f}")     # test R² (~1.0000)
print(f"NRMSE = {esn.nrmse():.6f}")  # test NRMSE (~0.001)
```

### Explicit (full control)

```python
import numpy as np
import hypercube_rc as hrc

signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)

esn = hrc.ESN(dim=7, seed=42)
esn.warmup(signal[:200])
esn.run(signal[200:-1])

targets = signal[201:]
esn.train(targets[:1400])

r2 = esn.r2(targets, start=1400)  # count defaults to all remaining
print(f"R² = {r2:.6f}")
```

---

## API Reference

### The `dim` parameter

`dim` controls the hypercube dimension. The reservoir has N = 2^dim neurons. Supported values: 5-16.

| dim  | Neurons   | Typical use |
|------|-----------|-------------|
| 5    | 32        | Fast prototyping, embedded |
| 6    | 64        | Light benchmarks |
| 7    | 128       | Standard benchmarks |
| 8    | 256       | Production, complex tasks |
| 9-16 | 512-65536 | Research, high-capacity tasks |

For dim 9+, reduce `output_fraction` to control readout input size (e.g., 0.25 for dim 10).

---

### ESN

Owns the full Reservoir → Readout pipeline. The core API covers construction,
batch training, prediction, and evaluation. Additional streaming/online methods
are available for advanced use (see [Streaming / Online Training](#streaming--online-training)).

```python
import hypercube_rc as hrc

# Construction
esn = hrc.ESN(dim=7)                                                    # defaults
esn = hrc.ESN(dim=7, seed=42, leak_rate=0.3)                            # custom config

# High-level pipeline (recommended)
esn.fit(signal, warmup=200)                     # warmup + run + train
esn.fit(inputs, targets=labels, warmup=200)     # explicit targets (multi-input)
esn.r2()                                        # test R² (no args after fit)
esn.nrmse()                                     # test NRMSE

# Low-level pipeline (full control)
esn.warmup(inputs)                # drive without recording
esn.run(inputs)                   # drive and collect states
esn.clear_states()                # clear collected data (keeps readout)
esn.train(targets)                # HCNN readout, default config
esn.train_cnn(targets, epochs=1000, lr_max=0.0015)  # HCNN with custom config

# Prediction & evaluation
esn.predict_raw(timestep)           # single continuous prediction
esn.predictions()                   # all predictions as ndarray
esn.r2(targets, start=1400)         # R² from index 1400 to end
esn.nrmse(targets, start, count)    # normalized RMSE
esn.accuracy(labels, start, count)  # classification accuracy

# State access
esn.selected_states()               # stride-selected states as ndarray
```

---

#### Construction

```python
ESN(dim, *, seed=0, spectral_radius=0.9, input_scaling=0.02,
    leak_rate=1.0, alpha=1.0, num_inputs=1, output_fraction=1.0)
```

Creates the reservoir and computes output selection parameters from `output_fraction`. The reservoir weights are generated and spectral-radius-rescaled at construction time.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | — | Hypercube dimension (5-16). N = 2^dim neurons. |
| `seed` | `int` | `0` | RNG seed for weight initialization. Every seed (including 0) produces a valid weight topology; different seeds yield measurably different performance. Use per-DIM seed surveys to find optimal seeds for your task. |
| `spectral_radius` | `float` | `0.9` | Target spectral radius. Scale-invariant across all dim values (vertex-transitive topology property). No per-size re-tuning needed. |
| `input_scaling` | `float` | `0.02` | Input weight magnitude, U(-input_scaling, +input_scaling). Scale-invariant across all dim values. |
| `leak_rate` | `float` | `1.0` | Leaky integrator coefficient. 1.0 = full replacement. < 1.0 adds smoothing. |
| `alpha` | `float` | `1.0` | Gain inside tanh: `tanh(alpha * sum)`. > 1.0 sharpens nonlinearity. |
| `num_inputs` | `int` | `1` | Number of input channels. Channel k drives every K-th vertex starting at offset k. |
| `output_fraction` | `float` | `1.0` | Fraction of N vertices used as readout features, in (0.0, 1.0]. |

---

#### High-Level Pipeline

##### `fit(inputs, targets=None, *, warmup=200, train_size=None, train_frac=None, horizon=1) → ESN`

One-call pipeline that performs warmup, run, train, and stores targets for zero-argument evaluation. Returns `self` for method chaining.

**Two modes:**

**Auto-target** (`targets=None`, single-input only): generates next-step prediction targets from the input signal, shifted by `horizon` steps.

```python
esn.fit(signal, warmup=200)                   # next-step, 70% train
esn.fit(signal, warmup=200, train_size=1400)  # next-step, explicit split
esn.fit(signal, warmup=200, horizon=5)        # 5-step-ahead prediction
```

**Explicit-target** (any `num_inputs`): uses the provided targets array directly. Required for multi-input ESN and classification tasks. `horizon` is ignored.

```python
# Multi-input: predict channel 0
esn.fit(inputs, targets=ch0[201:], warmup=200)

# Classification
esn.fit(signal, targets=labels, warmup=200)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `ndarray` | — | Input signal. Shape `(steps,)` or `(steps, num_inputs)`. |
| `targets` | `ndarray` | `None` | One target per collected state. Required for multi-input. |
| `warmup` | `int` | `200` | Timesteps for transient washout. |
| `train_size` | `int` | `None` | Training samples. Mutually exclusive with `train_frac`. |
| `train_frac` | `float` | `None` | Training fraction. Default 0.7 when neither is given. |
| `horizon` | `int` | `1` | Auto-target prediction horizon. Ignored with explicit targets. |

**After `fit()`**, call `r2()`, `nrmse()`, or `accuracy()` with no arguments to evaluate the held-out test portion:

```python
esn.fit(signal, warmup=200)
print(esn.r2())       # test R²
print(esn.nrmse())    # test NRMSE
print(esn.train_size) # number of training samples
print(esn.test_size)  # number of test samples
```

---

#### Low-Level Pipeline

The methods below give full control over each step. Use these for multi-step workflows, streaming, or when `fit()` doesn't match your use case.

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

##### `train(targets)`

Train the HCNN readout with default config using `len(targets)` training samples from the start of the collected states.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `targets` | `ndarray` | — | Target values, shape `(train_size,)`. |

**Notes:**
- Raises `ValueError` if `len(targets) > num_collected`.
- Calling `train()` again replaces the previous solution entirely.
- For full control over CNN config, use `train_cnn()` instead.

##### `train_cnn(targets, **kwargs)`

Train the HCNN readout with explicit config. All CNN hyperparameters are keyword arguments.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `targets` | `ndarray` | — | Target values. Regression: `(train_size,)` or `(train_size * num_outputs,)`. Classification: `(train_size,)` float class labels. |
| `num_outputs` | `int` | `1` | Number of outputs (classes for classification, targets for regression). |
| `task` | `str` | `"regression"` | `"regression"` or `"classification"`. |
| `num_layers` | `int` | `0` | Conv+Pool pairs. 0 = auto: min(DIM-2, 2). |
| `conv_channels` | `int` | `16` | Base channels (doubles per layer). |
| `epochs` | `int` | `200` | Training epochs. |
| `batch_size` | `int` | `32` | Mini-batch size. |
| `lr_max` | `float` | `0.005` | Cosine annealing peak learning rate. |
| `lr_min_frac` | `float` | `0.1` | Floor = lr_max * lr_min_frac. |
| `lr_decay_epochs` | `int` | `0` | Cosine decay horizon. 0 = use `epochs`. |
| `weight_decay` | `float` | `0.0` | L2 weight decay. |
| `seed` | `int` | `42` | CNN weight initialization seed. |
| `verbose` | `bool` | `False` | Print per-epoch learning rate. |

---

#### Prediction and Evaluation

##### `predict_raw(timestep) → float`

Return the raw continuous prediction for a single collected timestep.

**Parameters:**
- `timestep` — Index into collected states, in [0, num_collected).

**Returns:** Continuous float prediction. For regression: the predicted value. For binary classification (single output): threshold at 0.0. For multi-class classification: a single logit (use `accuracy()` for evaluation, which applies argmax internally).

---

##### `predictions() → ndarray`

Return predictions for all collected timesteps as a 1D float32 array of shape `(num_collected,)`.

---

##### `r2(targets=None, start=None, count=None) → float`

Compute R-squared (coefficient of determination) on a slice of collected states.

```
R² = 1 - SS_res / SS_tot
```

**Calling conventions:**

```python
esn.r2()                       # after fit(): test R² (uses stored targets)
esn.r2(targets)                # all collected states
esn.r2(targets, start=1400)    # from index 1400 to end
esn.r2(targets, start=0, count=1400)  # first 1400 states only
```

**Parameters:**
- `targets` — Target array, index-aligned with collected states (`targets[i]` is the target for collected state `i`). If omitted, uses targets stored by `fit()`.
- `start` — First timestep index. Default: 0, or `train_size` after `fit()`.
- `count` — Number of timesteps to evaluate. Default: all remaining from `start`.

**Returns:** R² value. 1.0 = perfect. 0.0 = predicts the mean. Can be negative.

> **Warning:** Do not slice the targets array before passing. The `start` parameter indexes into **both** the internal feature buffer and the target array simultaneously. Slicing targets shifts the alignment and produces wrong results silently. Use the `start` parameter instead.
>
> ```python
> esn.r2(targets, start=1400)      # CORRECT
> esn.r2(targets[1400:])           # WRONG — evaluates training features against test targets
> ```

---

##### `nrmse(targets=None, start=None, count=None) → float`

Compute Normalized Root Mean Squared Error on a slice of collected states.

```
NRMSE = sqrt(MSE) / sqrt(Var(targets))
```

**Parameters:** Same conventions as `r2()`.

**Returns:** NRMSE value. 0.0 = perfect. 1.0 = as bad as predicting the mean.

---

##### `accuracy(labels=None, start=None, count=None) → float`

Compute classification accuracy on a slice of collected states.

- **Multi-class** (`num_outputs > 1`): computes argmax over class logits and compares to the label. Labels are float class indices (0.0, 1.0, 2.0, ...).
- **Binary** (`num_outputs == 1`): thresholds prediction at 0.0 and compares sign to label. Labels are {-1.0, +1.0}.

**Parameters:** Same conventions as `r2()`.

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
| `num_outputs` | `int` | Number of readout outputs (after training). |
| `num_inputs` | `int` | Number of input channels. |
| `output_fraction` | `float` | Fraction of vertices used as readout input. |
| `num_output_verts` | `int` | Number of selected output vertices M. |
| `alpha` | `float` | Tanh gain parameter. |
| `seed` | `int` | RNG seed used to initialize reservoir weights. |
| `spectral_radius` | `float` | Target spectral radius. |
| `leak_rate` | `float` | Leaky integrator coefficient. |
| `input_scaling` | `float` | Input weight magnitude. |
| `train_size` | `int \| None` | Training samples from `fit()`, or None. |
| `test_size` | `int \| None` | Test samples from `fit()`, or None. |

---

#### Streaming / Online Training

For applications where data arrives continuously. These methods mirror the C++
streaming API (see `docs/CPP_SDK.md` for detailed parameter documentation).

| Method | Description |
|--------|-------------|
| `init_online(warmup_inputs, **kwargs)` | Initialize HCNN for streaming. Runs warmup, computes standardization, builds CNN. Call before `train_live_*`. |
| `compute_target_centering(targets)` | Set per-output target means for online regression (matching batch centering behavior). |
| `train_live_step(target_class, lr)` | Single-sample online gradient step (classification). |
| `train_live_batch(states, targets, lr)` | Mini-batch online gradient step (classification). |
| `train_live_step_regression(target, lr)` | Single-sample online gradient step (regression). |
| `train_live_batch_regression(states, targets, lr)` | Mini-batch online gradient step (regression). |
| `copy_live_state()` | Copy current subsampled reservoir state for batch accumulation. Returns `(num_output_verts,)` array. |
| `predict_live_raw()` | Scalar prediction from current live state. |
| `predict_live_raw_multi()` | Multi-output prediction from current live state. Returns `(num_outputs,)` array. |

#### Reservoir State Management

| Method | Description |
|--------|-------------|
| `reset_reservoir_only()` | Zero the reservoir state; collected states and trained readout are preserved. For episodic tasks. |
| `save_reservoir_state()` | Snapshot internal state. Returns `(state, output)` tuple of `(N,)` arrays. |
| `restore_reservoir_state(state, output)` | Restore a previously saved snapshot. |

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

## Data Types

The C++ reservoir operates entirely in **float32** — weights, states, features, and readout. This is by design: the tanh nonlinearity squashes values to [-1, 1], weights are random, and the topology's inherent noise far exceeds float32 rounding error. Float64 would produce identical results.

All input arrays (signals, targets, labels) are automatically converted to C-contiguous float32 before being passed to C++. NumPy defaults to float64, so this conversion happens silently on most calls. No precision is lost in practice.

If you want to avoid the conversion overhead on hot paths, pre-cast your arrays:

```python
signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
```

---

## Error Handling

The Python bindings validate arguments at the boundary and raise clear exceptions:

- **`ValueError`** — invalid `dim` (not 5-16), `train_size > num_collected`, or input array size not divisible by `num_inputs`.
- **`IndexError`** — `predict_raw(timestep)` with `timestep >= num_collected`, or `r2`/`nrmse`/`accuracy` with `start + count > num_collected`.

These checks happen before calling into C++, so you get a Python traceback instead of a crash.

---

## Model Persistence

Trained ESN models can be saved to disk and restored without retraining. The reservoir weights are deterministic from the seed, so only the config and trained readout are persisted. Files are compact (typically < 1 MB).

#### `esn.save(path)`

Save the trained ESN to a file (standard Python pickle).

```python
esn = hrc.ESN(dim=7, seed=42)
esn.fit(signal, warmup=200)
esn.save("model.pkl")
```

#### `ESN.load(path) -> ESN`

Load a saved ESN. Returns a new ESN with the trained readout intact and zero collected states.

```python
loaded = hrc.ESN.load("model.pkl")
loaded.warmup(new_signal[:200])
loaded.run(new_signal[200:])
preds = loaded.predictions()
```

#### Pickle support

ESN objects support `pickle.dumps()` / `pickle.loads()` directly:

```python
import pickle
data = pickle.dumps(esn)
restored = pickle.loads(data)
```

#### What is and isn't saved

| Saved | Not saved |
|-------|-----------|
| All constructor parameters (dim, seed, spectral_radius, etc.) | Collected states (regenerate with `warmup()` + `run()`) |
| Trained readout weights, bias, and standardization stats | Cached features |
| Readout type | `fit()` targets and train/test split |

---

## Limitations

- **No scikit-learn compatibility.** The ESN is a temporal pipeline (input order matters, warmup required, states accumulate sequentially), not a static feature→label model. The sklearn estimator protocol assumes i.i.d. samples and row-shuffled cross-validation, which would destroy the temporal structure.
- **No raw buffer access.** The C++ SDK exposes raw state buffers for diagnostics. The Python bindings provide `selected_states()` and `predictions()` instead, which return NumPy arrays.

---

## Dependencies

**Runtime:** NumPy >= 1.21

**Build time:** scikit-build-core >= 0.10, pybind11 >= 2.13, C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+), CMake 3.20+, HypercubeCNN (sibling static library, linked at build time)

Pre-built wheels bundle HypercubeCNN statically, so no action is needed when installing from PyPI. When building from source, HypercubeCNN must be available at `../HypercubeCNN` relative to the HypercubeRC source tree (override `HCNN_DIR` in `python/CMakeLists.txt` if elsewhere).
