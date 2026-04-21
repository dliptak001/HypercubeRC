# HypercubeRC C++ SDK

Static C++ library for reservoir computing on Boolean hypercube graphs.

## Contents

- [What's in the SDK](#whats-in-the-sdk)
- [Building from source](#building-from-source)
- [Using the SDK](#using-the-sdk)
  - [CMake FetchContent (recommended)](#cmake-fetchcontent-recommended)
  - [Installed SDK (find_package)](#installed-sdk-find_package)
- [API Reference](#api-reference)
  - [Template parameter: DIM](#template-parameter-dim)
  - [Enums](#enums)
  - [ReservoirConfig](#reservoirconfig)
  - [CNNReadoutConfig](#cnnreadoutconfig)
  - [ESN\<DIM\>](#esndim)
- [Dependencies](#dependencies)

## What's in the SDK

After installation, the SDK contains:

```
<prefix>/
  include/HypercubeRC/
    ESN.h              -- The public API (the only header consumers include)
    Reservoir.h        -- Internal: included by ESN.h
    TranslationLayer.h -- Internal: included by ESN.h
    RidgeRegression.h  -- Internal: included by ESN.h
    CNNReadout.h       -- Internal: included by ESN.h (HCNN readout config)
  lib/
    libHypercubeRCCore.a
  lib/cmake/HypercubeRC/
    HypercubeRCConfig.cmake
    HypercubeRCTargets.cmake
    HypercubeRCConfigVersion.cmake
```

Consumers include `<HypercubeRC/ESN.h>` and link against `HypercubeRC::HypercubeRCCore`. The internal headers are present because ESN.h includes them, but there is no need to include or interact with them directly.

The SDK depends on a second static library — **HypercubeCNN** — that provides the convolutional readout. HypercubeRCCore transitively links to it, so consumers don't need to reference it explicitly, but it must be buildable at configure time. See [Dependencies](#dependencies).

## Building from source

Requirements: C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+), CMake 4.1+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk
```

## Using the SDK

### CMake FetchContent (recommended)

The simplest way to use HypercubeRC in a CMake project. No installation, no
manual downloads -- CMake pulls the source from GitHub and builds it alongside
your project.

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyApp)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
FetchContent_Declare(
    HypercubeRC
    GIT_REPOSITORY https://github.com/dliptak001/HypercubeRC.git
    GIT_TAG        v0.2.0
)
FetchContent_MakeAvailable(HypercubeRC)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeRCCore)
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Pin `GIT_TAG` to a release tag (e.g., `v0.2.0`) for reproducible builds.
Include paths are set automatically -- just `#include "ESN.h"`.

### Installed SDK (find_package)

If you prefer to install the library once and link against it:

```bash
# Build and install
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk
```

```cmake
cmake_minimum_required(VERSION 4.1)
project(MyApp)

set(CMAKE_CXX_STANDARD 23)

find_package(HypercubeRC REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeRC::HypercubeRCCore)
```

Configure with the SDK path:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/sdk
cmake --build build
```

### Minimal example (Ridge readout)

```cpp
#include <HypercubeRC/ESN.h>
#include <cmath>
#include <vector>
#include <iostream>

int main()
{
    constexpr size_t DIM = 7;         // 2^7 = 128 neurons
    constexpr size_t warmup = 200;
    constexpr size_t collect = 2000;

    // Generate a sine wave
    std::vector<float> signal(warmup + collect + 1);
    for (size_t t = 0; t < signal.size(); ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    // Create ESN with default config
    ReservoirConfig cfg;
    cfg.seed = 42;
    ESN<DIM> esn(cfg);  // defaults: Ridge readout, Translated features

    // Drive and train
    esn.Warmup(signal.data(), warmup);
    esn.Run(signal.data() + warmup, collect);

    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + 1];  // predict next value

    size_t train_size = 1400;
    size_t test_size = collect - train_size;

    esn.Train(targets.data(), train_size);

    double r2 = esn.R2(targets.data(), train_size, test_size);
    std::cout << "R2: " << r2 << "\n";  // ~1.0000

    return 0;
}
```

### HCNN readout example

The HCNN (HypercubeCNN) readout replaces linear regression with a learned
convolutional network operating directly on raw reservoir state. It always
uses `FeatureMode::Raw` (the translation layer is bypassed) and requires
a `CNNReadoutConfig` for training.

```cpp
#include <HypercubeRC/ESN.h>
#include <cmath>
#include <vector>
#include <iostream>

int main()
{
    constexpr size_t DIM = 7;
    constexpr size_t warmup = 200;
    constexpr size_t collect = 2000;

    std::vector<float> signal(warmup + collect + 1);
    for (size_t t = 0; t < signal.size(); ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    ReservoirConfig cfg;
    cfg.seed = 42;
    cfg.output_fraction = 1.0f;  // HCNN operates on all N vertices
    ESN<DIM> esn(cfg, ReadoutType::HCNN);  // FeatureMode::Raw is forced

    esn.Warmup(signal.data(), warmup);
    esn.Run(signal.data() + warmup, collect);

    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + 1];

    size_t train_size = 1400;
    size_t test_size = collect - train_size;

    // HCNN uses its own Train overload that takes a CNNReadoutConfig.
    CNNReadoutConfig cnn_cfg;
    cnn_cfg.task = HCNNTask::Regression;
    cnn_cfg.num_outputs = 1;
    cnn_cfg.epochs = 25;           // HCNN saturates fast on structured signals
    cnn_cfg.batch_size = 128;
    cnn_cfg.lr_max = 0.003f;       // keep <= 0.003 to avoid NaN divergence
    esn.Train(targets.data(), train_size, cnn_cfg);

    double r2 = esn.R2(targets.data(), train_size, test_size);
    std::cout << "HCNN R2: " << r2 << "\n";

    return 0;
}
```

---

## API Reference

### Template parameter: DIM

`DIM` is a compile-time template parameter controlling the hypercube dimension. The reservoir has N = 2^DIM neurons. The library provides explicit template instantiations for **DIM 5-16**.

| DIM   | Neurons     | Typical use |
|-------|-------------|-------------|
| 5     | 32          | Fast prototyping, embedded |
| 6     | 64          | Light benchmarks |
| 7     | 128         | Standard benchmarks |
| 8     | 256         | Production, complex tasks |
| 9-12  | 512-4096    | Research, high-capacity tasks |
| 13-16 | 8192-65536  | Large-scale research only — Ridge O(M²) memory becomes prohibitive without `output_fraction` reduction |

For DIM 9+, reduce `output_fraction` to control Ridge readout cost (e.g., 0.25 for DIM 10). HCNN's cost grows roughly linearly in N rather than quadratically, so it scales more gracefully to larger DIM.

### Enums

#### `ReadoutType`

| Value | Description |
|-------|-------------|
| `Ridge` | Closed-form Ridge regression. Deterministic, fast, optimal for the given regularization. Default. |
| `HCNN` | Learned convolutional readout (HypercubeCNN). Operates directly on raw N-vertex state, bypassing the translation layer. Supports multi-output regression and multi-class classification. Trained via the `Train(targets, train_size, CNNReadoutConfig)` overload. See [CNNReadoutConfig](#cnnreadoutconfig). |

#### `FeatureMode`

| Value | Description |
|-------|-------------|
| `Translated` | Expands M selected states into 2.5M features via [x \| x^2 \| x\*x_antipodal]. Reduces NRMSE by 20-70% on standard benchmarks. Default for Ridge. |
| `Raw` | Uses M selected states directly. Fewer features, faster computation, sufficient for simple tasks. |

**Note:** When `ReadoutType::HCNN` is selected, the ESN constructor forces `FeatureMode::Raw` regardless of what you pass. HCNN operates on raw N-vertex state and has its own internal standardization — it never uses the translation layer.

#### `HCNNTask`

Task head for the HCNN readout. Declared in `CNNReadout.h`.

| Value | Description |
|-------|-------------|
| `Regression` | MSE loss, de-centered predictions. `num_outputs` sets the number of regression targets. |
| `Classification` | Softmax + cross-entropy loss. `num_outputs` sets the number of classes. Targets are float class indices; predictions are raw logits (use `argmax` or HCNN's internal `PredictClass`). |

---

### ReservoirConfig

Configuration struct for reservoir construction. All fields have scale-invariant defaults that work across all DIM values without re-tuning.

```cpp
struct ReservoirConfig
{
    uint64_t seed             = 0;
    float    alpha            = 1.0f;
    float    spectral_radius  = 0.9f;
    float    leak_rate        = 1.0f;
    float    input_scaling    = 0.02f;
    size_t   num_inputs       = 1;
    float    output_fraction  = 1.0f;
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | `uint64_t` | `0` | RNG seed for weight initialization. Every seed (including 0) produces a valid weight topology; different seeds yield measurably different performance. Use [SeedSurvey](SeedSurvey.md) to find optimal seeds. |
| `alpha` | `float` | `1.0` | Gain applied inside the tanh activation: `tanh(alpha * sum)`. Values > 1.0 sharpen the nonlinearity; < 1.0 linearize it. |
| `spectral_radius` | `float` | `0.9` | Target spectral norm for recurrent weight matrix. Controls the echo state property -- how quickly past inputs fade. Scale-invariant across all DIM values (vertex-transitive topology property). |
| `leak_rate` | `float` | `1.0` | Leaky integrator coefficient. `state = (1 - leak_rate) * old_state + leak_rate * activation`. At 1.0 (default), each step fully replaces state. Values < 1.0 add temporal smoothing. |
| `input_scaling` | `float` | `0.02` | Magnitude of input weights, drawn from U(-input_scaling, +input_scaling). Scale-invariant across all DIM values. |
| `num_inputs` | `size_t` | `1` | Number of input channels. In multi-input mode (K channels), channel k drives every K-th vertex starting at offset k (stride-interleaved). |
| `output_fraction` | `float` | `1.0` | Fraction of N vertices used as readout features, in range (0.0, 1.0] (0.0 is not valid). At 0.5, a stride-selected subset of N/2 vertices is used, reducing Ridge readout cost by ~4x (quadratic in feature count). HCNN always consumes all N vertices regardless of this value. |

---

### CNNReadoutConfig

Configuration struct for the HCNN readout. Only used by the `Train(targets, train_size, CNNReadoutConfig)` overload when `ReadoutType::HCNN` is selected.

```cpp
struct CNNReadoutConfig {
    int num_outputs    = 1;
    HCNNTask task      = HCNNTask::Regression;
    int num_layers     = 0;        // 0 = auto: min(DIM - 3, 4)
    int conv_channels  = 16;       // doubles per layer
    int epochs         = 200;
    int batch_size     = 32;
    float lr_max       = 0.005f;
    float lr_min_frac  = 0.1f;
    float weight_decay = 0.0f;
    unsigned seed      = 42;
    bool verbose       = false;
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_outputs` | `int` | `1` | Number of output neurons. For regression: number of targets. For classification: number of classes. |
| `task` | `HCNNTask` | `Regression` | Task head. See [HCNNTask](#hcnntask). |
| `num_layers` | `int` | `0` (auto) | Number of Conv+Pool pairs. `0` auto-computes `min(DIM - 3, 4)` — each Pool halves the hypercube dimension, so the stack depth is capped by `DIM - 3` (HCNNConv requires DIM ≥ 3). |
| `conv_channels` | `int` | `16` | Base channel count for the first Conv layer. Channels double per layer (16, 32, 64, 128 for a 4-layer stack). |
| `epochs` | `int` | `200` | Training epochs. HCNN saturates very quickly on structured reservoir state — 25 epochs is typically enough for examples; benchmarks use 300. |
| `batch_size` | `int` | `32` | Mini-batch size. Use 128 on CPUs with multiple cores to saturate SIMD + threading. |
| `lr_max` | `float` | `0.005` | Peak learning rate for cosine annealing. **Keep `lr_max <= 0.003` to avoid weight divergence into denormal/NaN territory, which collapses CPU throughput.** |
| `lr_min_frac` | `float` | `0.1` | Cosine schedule floor as fraction of `lr_max`. Effective `lr_min = lr_max * lr_min_frac`. |
| `weight_decay` | `float` | `0.0` | L2 weight decay applied by the Adam optimizer. |
| `seed` | `unsigned` | `42` | Seed for weight initialization. |
| `verbose` | `bool` | `false` | Print per-epoch training accuracy (classification only). |

**Architecture auto-sizing table:**

| DIM  | Auto layers | Channels             | Final DIM |
|------|-------------|----------------------|-----------|
| 5    | 1           | 16                   | 4         |
| 6    | 2           | 16, 32               | 4         |
| 7    | 3           | 16, 32, 64           | 4         |
| 8-16 | 4 (cap)     | 16, 32, 64, 128      | DIM − 4   |

See `readout/CNNReadout.md` for the full design notes and benchmark data.

---

### ESN\<DIM\>

The complete API. Owns the full Reservoir -> [Translation] -> Readout pipeline (translation is bypassed when `ReadoutType::HCNN` is selected).

```cpp
// Construction
ESN<DIM>(cfg);                                              // Ridge + Translated (defaults)
ESN<DIM>(cfg, ReadoutType::Ridge, FeatureMode::Raw);        // explicit selection
ESN<DIM>(cfg, ReadoutType::HCNN);                           // HCNN (forces FeatureMode::Raw)

// Reservoir driving
esn.Warmup(inputs, num_steps);              // drive without recording
esn.Run(inputs, num_steps);                 // drive and collect states
esn.ClearStates();                          // clear collected data (keeps readout)

// Training
esn.Train(targets, train_size);                  // default parameters
esn.Train(targets, train_size, lambda);          // Ridge: custom regularization
esn.Train(targets, train_size, cnn_cfg);         // HCNN: custom CNN config

// Prediction & evaluation
esn.PredictRaw(timestep);                   // scalar prediction (all readouts)
esn.PredictRaw(timestep, output);           // multi-output prediction (HCNN)
esn.R2(targets, start, count);              // R-squared (averaged across outputs)
esn.NRMSE(targets, start, count);           // normalized RMSE
esn.Accuracy(labels, start, count);         // classification accuracy
esn.NumOutputs();                           // 1 for Ridge, config for HCNN

// State & feature access
esn.States();                               // raw N-dim state buffer
esn.SelectedStates();                       // stride-selected M-dim states
esn.Features();                             // cached feature buffer (Ridge)
esn.EnsureFeatures();                       // force feature computation
esn.NumFeatures();                          // features per timestep (N for HCNN)
esn.NumCollected();                         // timesteps recorded
```

#### Construction

```cpp
explicit ESN(const ReservoirConfig& cfg,
             ReadoutType readout_type = ReadoutType::Ridge,
             FeatureMode feature_mode = FeatureMode::Translated);
```

Creates the reservoir from `cfg`, initializes the selected readout type, and computes output selection parameters from `cfg.output_fraction`. The reservoir weights are generated and spectral-radius-rescaled at construction time.

**Parameters:**
- `cfg` -- Reservoir configuration. See [ReservoirConfig](#reservoirconfig).
- `readout_type` -- Which readout to use. Default: `ReadoutType::Ridge`.
- `feature_mode` -- Whether to apply the translation layer. Default: `FeatureMode::Translated`. **Forced to `FeatureMode::Raw` when `readout_type == ReadoutType::HCNN`** regardless of what you pass.

---

#### Reservoir Driving

##### `Warmup`

```cpp
void Warmup(const float* inputs, size_t num_steps);
```

Drives the reservoir for `num_steps` timesteps without recording state. Use this to wash out the reservoir's initial transient (zero state) before collecting data for training.

**Parameters:**
- `inputs` -- Pointer to `num_steps * num_inputs` floats, row-major. Each timestep has `num_inputs` consecutive values (one per channel). When `num_inputs == 1` (default), this is simply `num_steps` scalars. Values are clamped internally to [-1, +1].
- `num_steps` -- Number of timesteps to drive. Typical: 100-500 depending on task.

**Notes:**
- Does not allocate memory or record states.
- The reservoir's internal state is updated in-place.

---

##### `Run`

```cpp
void Run(const float* inputs, size_t num_steps);
```

Drives the reservoir for `num_steps` timesteps, recording the full N-dimensional state vector at each step. States are appended to the internal buffer -- multiple `Run()` calls accumulate.

**Parameters:**
- `inputs` -- Pointer to `num_steps * num_inputs` floats, row-major. Same layout as `Warmup()`.
- `num_steps` -- Number of timesteps to drive and record.

**Notes:**
- Allocates/extends the internal state buffer to hold `(existing + num_steps) * N` floats.
- After `Run()`, the collected states are available via `States()` and can be used for training and evaluation.
- Features for new states are not computed immediately -- they are computed lazily when next accessed (via `EnsureFeatures()`, or implicitly by `Train()`, `R2()`, etc.). Previously computed features remain valid.

---

##### `ClearStates`

```cpp
void ClearStates();
```

Clears all collected states and cached features. The reservoir's live internal state is **not** reset -- it retains its current activation. The trained readout is also preserved.

Use this between independent sequences: clear the collected data, then `Warmup()` + `Run()` on a new input sequence without rebuilding the ESN.

---

#### Training

##### `Train` (default parameters)

```cpp
void Train(const float* targets, size_t train_size);
```

Trains the readout on the first `train_size` collected states using default parameters for the selected readout type.

- **Ridge** (default): lambda = 1.0
- **HCNN**: delegates to the `Train(targets, train_size, CNNReadoutConfig{})` overload below with default-constructed config.

**Parameters:**
- `targets` -- Pointer to target values. Must have at least `train_size` elements (or `train_size * num_outputs` for HCNN multi-output regression). For Ridge regression: continuous float values. For Ridge classification: {-1.0, +1.0}. For HCNN classification: float class indices.
- `train_size` -- Number of samples to train on, starting from collected state index 0.

**Notes:**
- Triggers feature computation (`EnsureFeatures()`) if not already done (Ridge only; HCNN reads raw state directly).
- Features are standardized internally by the readout (zero mean, unit variance).
- For Ridge, calling `Train()` again replaces the previous solution entirely.
- For HCNN, calling `Train()` again rebuilds the network from scratch.

---

##### `Train` (Ridge with custom lambda)

```cpp
void Train(const float* targets, size_t train_size, double lambda);
```

Trains the Ridge readout with a custom regularization strength. **Asserts that `readout_type` is `Ridge`.**

**Parameters:**
- `targets` -- Target values, at least `train_size` elements.
- `train_size` -- Number of training samples from state index 0.
- `lambda` -- Regularization strength. Larger values increase bias but reduce overfitting. Typical range: 0.01 to 100.0. Default (in the overload above) is 1.0.

---

##### `Train` (HCNN with CNNReadoutConfig)

```cpp
void Train(const float* targets, size_t train_size,
           const CNNReadoutConfig& config);
```

Trains the HCNN readout on raw reservoir state, bypassing the translation layer. **Asserts that `readout_type` is `HCNN`.**

**Parameters:**
- `targets` -- Target values. Layout depends on `config.task`:
  - **Regression:** `train_size * config.num_outputs` floats, row-major (one row per sample, one column per output).
  - **Classification:** `train_size` floats where each value is a float class index (cast from integer class label).
- `train_size` -- Number of training samples from collected state index 0.
- `config` -- Architecture and training hyperparameters. See [CNNReadoutConfig](#cnnreadoutconfig).

**Notes:**
- HCNN reads directly from `States()` (raw N-vertex state) — the feature pipeline is not touched.
- Per-vertex input standardization (mean/std) is computed from the training set and stored.
- For regression, per-output target centering is applied internally; predictions are de-centered at inference.
- For classification, softmax+CE loss; predictions from `PredictRaw(timestep, out)` are raw logits (apply argmax to get a class).
- Training is single-task — a second `Train()` call rebuilds the network from scratch with the new config.

---

---

#### Prediction and Evaluation

##### `PredictRaw` (scalar)

```cpp
[[nodiscard]] float PredictRaw(size_t timestep) const;
```

Returns the raw (continuous) prediction for a single collected timestep. Valid for Linear/Ridge and for HCNN when `num_outputs == 1`.

**Parameters:**
- `timestep` -- Index into collected states, in [0, NumCollected()). Asserts in range.

**Returns:** Continuous float prediction. For Linear/Ridge classification, threshold at 0.0 to get a class label, or use `Accuracy()` which handles thresholding internally. For HCNN with `num_outputs > 1`, use the multi-output overload below instead — the scalar form asserts.

---

##### `PredictRaw` (multi-output)

```cpp
void PredictRaw(size_t timestep, float* output) const;
```

Writes `NumOutputs()` floats to `output` for a single collected timestep. Use this for HCNN regression with multiple targets or HCNN classification with multiple classes (in which case the outputs are raw logits — apply argmax to get a predicted class). For Linear/Ridge this writes exactly one float to `output[0]`.

**Parameters:**
- `timestep` -- Index into collected states, in [0, NumCollected()).
- `output` -- Caller-provided buffer of at least `NumOutputs()` floats.

---

##### `R2`

```cpp
[[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;
```

Computes R-squared (coefficient of determination) on a slice of collected states.

```
R² = 1 - SS_res / SS_tot
```

**Parameters:**
- `targets` -- Full target array.
  - **Linear/Ridge:** slice `targets[start .. start+count)` is used (one float per sample).
  - **HCNN:** row-major `targets[start * NumOutputs() .. (start+count) * NumOutputs())` — one row of `NumOutputs()` floats per sample.
- `start` -- First timestep index (indexes into both the feature buffer and the target array).
- `count` -- Number of timesteps to evaluate.

**Returns:** R² value. 1.0 = perfect prediction. 0.0 = predicts the mean. Can be negative (worse than predicting the mean). For HCNN multi-output, returns the **average R² across outputs**.

**Notes:**
- Asserts `start + count <= NumCollected()`.
- Triggers `EnsureFeatures()` if needed (Linear/Ridge only).

**Typical usage (train/test split):**

```cpp
esn.Train(targets, train_size);
double test_r2 = esn.R2(targets, train_size, test_size);
```

Here `start = train_size` skips the training data and evaluates on the held-out test portion.

---

##### `NRMSE`

```cpp
[[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;
```

Computes Normalized Root Mean Squared Error on a slice of collected states.

```
NRMSE = sqrt(MSE) / sqrt(variance of targets)
```

**Parameters:**
- `targets` -- Full target array. Same layout rules as [`R2`](#r2):
  - **Linear/Ridge:** one float per sample at `targets[start .. start+count)`.
  - **HCNN:** row-major `NumOutputs()` floats per sample at `targets[start * NumOutputs() ..]`.
- `start` -- First timestep index.
- `count` -- Number of timesteps to evaluate.

**Returns:** NRMSE value. 0.0 = perfect. 1.0 = as bad as predicting the mean. Returns infinity if target variance < 1e-12 (constant signal). For HCNN multi-output, returns the **average NRMSE across outputs**.

**Notes:**
- Returns 0.0 if `count` is 0.
- The standard metric for reservoir computing benchmarks (Mackey-Glass, NARMA-10).

---

##### `Accuracy`

```cpp
[[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;
```

Computes classification accuracy on a slice of collected states.

**Parameters:**
- `labels` -- Full label array at `labels[start .. start+count)` (one float per sample regardless of readout type).
  - **Linear/Ridge:** labels are {-1.0, +1.0}. Predictions are thresholded at 0.0 (positive -> +1, negative -> -1).
  - **HCNN binary** (`num_outputs == 1`): labels are {-1.0, +1.0}, same thresholding.
  - **HCNN multi-class** (`num_outputs > 1`): labels are float class indices (`0.0`, `1.0`, ...). Predictions are argmax over the logit vector.
- `start` -- First timestep index.
- `count` -- Number of timesteps to evaluate.

**Returns:** Fraction of correct predictions in [0.0, 1.0].

---

#### State and Feature Access

##### `States`

```cpp
[[nodiscard]] const float* States() const;
```

Returns a pointer to the raw collected state buffer. Layout: `num_collected * N` floats, row-major (one N-dimensional state vector per timestep).

---

##### `SelectedStates`

```cpp
[[nodiscard]] std::vector<float> SelectedStates() const;
```

Extracts stride-selected vertices from all collected states. Returns a new vector of `num_collected * M` floats, where M = `NumOutputVerts()`. Vertices are selected by stride: indices 0, stride, 2*stride, ...

---

##### `Features`

```cpp
[[nodiscard]] const float* Features() const;
```

Returns a pointer to the cached feature buffer. Layout: `num_collected * NumFeatures()` floats, row-major.

**Note:** Returns whatever has been computed so far. Call `EnsureFeatures()` first if you need features for all collected states.

---

##### `EnsureFeatures`

```cpp
void EnsureFeatures() const;
```

Computes features for any collected states that haven't been processed yet. Incremental -- only processes new states since the last call. Called automatically by `Train()`, `PredictRaw()`, `R2()`, `NRMSE()`, and `Accuracy()`.

In `Translated` mode, applies the translation transform (x, x^2, x*x_antipodal) to stride-selected vertices. In `Raw` mode, extracts stride-selected vertices without transformation.

---

##### `NumFeatures`

```cpp
[[nodiscard]] size_t NumFeatures() const;
```

Returns the number of features per timestep.
- **Translated mode:** `M + M + M/2` = 2.5M, where M = `NumOutputVerts()`.
- **Raw mode:** M.

---

##### Other Accessors

| Method | Returns | Description |
|--------|---------|-------------|
| `NumCollected()` | `size_t` | Number of timesteps recorded by `Run()`. |
| `NumOutputs()` | `size_t` | `1` for Linear/Ridge; `CNNReadoutConfig::num_outputs` for trained HCNN. |
| `OutputFraction()` | `float` | The `output_fraction` from config. |
| `OutputStride()` | `size_t` | Stride used for vertex selection: `max(1, N / M)`. |
| `NumOutputVerts()` | `size_t` | Number of selected vertices M = ceil(N / stride). |
| `GetReadoutType()` | `ReadoutType` | Readout type selected at construction. |
| `GetFeatureMode()` | `FeatureMode` | Feature mode selected at construction (forced to `Raw` for HCNN). |
| `GetAlpha()` | `float` | The tanh gain alpha from the reservoir config. |
| `NumInputs()` | `size_t` | Number of input channels from config. |
| `GetConfig()` | `ReservoirConfig` | Full config used to construct this ESN (for serialization). |

---

##### Readout State Access

The ESN exposes its trained readout state for serialization. The reservoir weights are deterministic from the seed, so only the config and readout state need to be saved.

**`ReadoutState` struct** (nested in `ESN<DIM>`):

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `std::vector<double>` | **Linear/Ridge:** learned weight vector. **HCNN:** opaque flattened blob of all conv kernels, biases, and dense-head weights (layout defined by `hcnn::HCNN::GetWeights` / `SetWeights`). Do not interpret the values — just round-trip them. |
| `bias` | `double` | **Linear/Ridge:** learned bias. **HCNN:** fallback value for per-output target centering when the full target-mean vector isn't round-tripped separately. |
| `feature_mean` | `std::vector<float>` | **Linear/Ridge:** per-feature mean from the 2.5M-feature standardization. **HCNN:** per-vertex mean from the raw N-vertex standardization. |
| `feature_scale` | `std::vector<float>` | **Linear/Ridge:** per-feature 1/std. **HCNN:** per-vertex 1/std. |
| `is_trained` | `bool` | True if the readout has been trained. |

| Method | Returns | Description |
|--------|---------|-------------|
| `GetReadoutState()` | `ReadoutState` | Extract trained readout for serialization. |
| `SetReadoutState(state)` | `void` | Restore a previously saved readout state. |

For HCNN, `SetReadoutState` reconstructs the full network architecture from the stored `CNNReadoutConfig` and injects the weight blob — no retraining needed. The stored ESN must be recreated with `ReadoutType::HCNN` and the same `DIM`; the HCNN config (num_outputs, num_layers, conv_channels) must also be set on the readout before `SetReadoutState` is called, which happens when you construct the ESN and invoke any HCNN training path, or when the consuming application owns its own `CNNReadoutConfig` serialization separately.

**Example: save and restore a trained model (Linear/Ridge)**

```cpp
// Save
auto cfg = esn.GetConfig();
auto state = esn.GetReadoutState();
// ... serialize cfg, state.weights, state.bias, state.feature_mean,
//     state.feature_scale, readout_type, feature_mode, DIM
//     using your preferred format (JSON, protobuf, binary, etc.)

// Restore
ESN<6> restored(cfg, readout_type, feature_mode);
restored.SetReadoutState(state);
// Ready to predict — no retraining needed.
// Call Warmup() + Run() on new data, then PredictRaw().
```

---

## Dependencies

HypercubeRC depends on a single external project:

**HypercubeCNN** — sibling library providing the hypercube convolutional network used by `CNNReadout` / `ReadoutType::HCNN`.

- Expected location: `../HypercubeCNN` relative to the HypercubeRC source tree (adjust `HCNN_DIR` in `CMakeLists.txt` if yours is elsewhere).
- Built as `libHypercubeCNNCore.a` inside its own `cmake-build-release` directory before HypercubeRC is configured.
- Public headers are re-exported through `HypercubeRCCore`'s include interface, so consumers of HypercubeRC do not need to add HypercubeCNN to their own link line — `target_link_libraries(my_app PRIVATE HypercubeRCCore)` pulls it in transitively.
- The HCNN headers used by HypercubeRC consumers are the ones re-exported by `CNNReadout.h` (forward-declared `hcnn::HCNN` via PIMPL); the full HCNN API is not part of the public HypercubeRC surface.

No other external dependencies beyond the C++ standard library.
