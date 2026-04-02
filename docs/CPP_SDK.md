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
    LinearReadout.h    -- Internal: included by ESN.h
    RidgeRegression.h  -- Internal: included by ESN.h
  lib/
    libHypercubeRCCore.a
  lib/cmake/HypercubeRC/
    HypercubeRCConfig.cmake
    HypercubeRCTargets.cmake
    HypercubeRCConfigVersion.cmake
```

Consumers include `<HypercubeRC/ESN.h>` and link against `HypercubeRC::HypercubeRCCore`. The internal headers are present because ESN.h includes them, but there is no need to include or interact with them directly.

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
    GIT_TAG        v0.1.1
)
FetchContent_MakeAvailable(HypercubeRC)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeRCCore)
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Pin `GIT_TAG` to a release tag (e.g., `v0.1.1`) for reproducible builds.
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

### Minimal example

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

---

## API Reference

### Template parameter: DIM

`DIM` is a compile-time template parameter controlling the hypercube dimension. The reservoir has N = 2^DIM neurons. The library provides explicit template instantiations for DIM 5-12.

| DIM | Neurons | Typical use |
|-----|---------|-------------|
| 5   | 32      | Fast prototyping, embedded |
| 6   | 64      | Light benchmarks |
| 7   | 128     | Standard benchmarks |
| 8   | 256     | Production, complex tasks |
| 9-12 | 512-4096 | Research, high-capacity tasks |

For DIM 9+, reduce `output_fraction` to control Ridge readout cost (e.g., 0.25 for DIM 10).

### Enums

#### `ReadoutType`

| Value | Description |
|-------|-------------|
| `Ridge` | Closed-form Ridge regression. Deterministic, fast, optimal for the given regularization. Default. |
| `Linear` | Online SGD with L2 decay and pocket selection. Supports streaming via `TrainIncremental()`. |

#### `FeatureMode`

| Value | Description |
|-------|-------------|
| `Translated` | Expands M selected states into 2.5M features via [x \| x^2 \| x\*x_antipodal]. Reduces NRMSE by 20-70% on standard benchmarks. Default. |
| `Raw` | Uses M selected states directly. Fewer features, faster computation, sufficient for simple tasks. |

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
| `output_fraction` | `float` | `1.0` | Fraction of N vertices used as readout features, in range (0.0, 1.0] (0.0 is not valid). At 0.5, a stride-selected subset of N/2 vertices is used, reducing Ridge readout cost by ~4x (quadratic in feature count). |

---

### ESN\<DIM\>

The complete API. Owns the full Reservoir -> Translation -> Readout pipeline.

```cpp
// Construction
ESN<DIM>(cfg);                                              // Ridge + Translated (defaults)
ESN<DIM>(cfg, ReadoutType::Linear, FeatureMode::Raw);       // explicit selection

// Reservoir driving
esn.Warmup(inputs, num_steps);              // drive without recording
esn.Run(inputs, num_steps);                 // drive and collect states
esn.ClearStates();                          // clear collected data (keeps readout)

// Training
esn.Train(targets, train_size);             // default parameters
esn.Train(targets, train_size, lambda);     // Ridge: custom regularization
esn.Train(targets, train_size, lr, epochs); // Linear: custom SGD
esn.TrainIncremental(targets, train_size, blend); // Linear: streaming update

// Prediction & evaluation
esn.PredictRaw(timestep);                   // single continuous prediction
esn.R2(targets, start, count);              // R-squared
esn.NRMSE(targets, start, count);           // normalized RMSE
esn.Accuracy(labels, start, count);         // classification accuracy

// State & feature access
esn.States();                               // raw N-dim state buffer
esn.SelectedStates();                       // stride-selected M-dim states
esn.Features();                             // cached feature buffer
esn.EnsureFeatures();                       // force feature computation
esn.NumFeatures();                          // features per timestep
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
- `feature_mode` -- Whether to apply the translation layer. Default: `FeatureMode::Translated`.

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
- **Linear**: lr = auto (1.0 / num_features), epochs = 200, weight_decay = 1e-4, lr_decay = 0.01

**Parameters:**
- `targets` -- Pointer to target values. Must have at least `train_size` elements. For regression: continuous float values. For classification: {-1.0, +1.0}.
- `train_size` -- Number of samples to train on, starting from collected state index 0.

**Notes:**
- Triggers feature computation (`EnsureFeatures()`) if not already done.
- Features are standardized internally by the readout (zero mean, unit variance).
- For Ridge, calling `Train()` again replaces the previous solution entirely.
- For Linear, calling `Train()` again retrains from scratch (use `TrainIncremental()` for streaming updates).

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

##### `Train` (Linear with custom SGD parameters)

```cpp
void Train(const float* targets, size_t train_size,
           float lr, size_t epochs,
           float weight_decay = 1e-4f, float lr_decay = 0.01f);
```

Trains the Linear readout with custom SGD parameters. **Asserts that `readout_type` is `Linear`.**

**Parameters:**
- `targets` -- Target values, at least `train_size` elements.
- `train_size` -- Number of training samples from state index 0.
- `lr` -- Learning rate. Pass 0.0 for auto-selection (1.0 / num_features).
- `epochs` -- Number of full passes over the training data.
- `weight_decay` -- L2 regularization coefficient. Applied per-update to all weights (not bias). Default: 1e-4.
- `lr_decay` -- Learning rate decay factor. Effective LR at epoch e = lr / (1 + lr_decay * e). Default: 0.01.

**Notes:**
- Uses pocket selection: the weight vector with the lowest training MSE across all epochs is retained as the final model.

---

##### `TrainIncremental`

```cpp
void TrainIncremental(const float* targets, size_t train_size,
                      float blend = 0.1f,
                      float lr = 0.0f, size_t epochs = 200,
                      float weight_decay = 1e-4f, float lr_decay = 0.01f);
```

Incrementally updates the Linear readout for streaming applications. **Asserts that `readout_type` is `Linear`.**

Trains a fresh model on the provided data, then blends it with the existing model:

```
W_updated = (1 - blend) * W_existing + blend * W_new
```

Feature standardization statistics (mean, scale) are blended the same way, allowing the model to track distribution drift over time.

**Parameters:**
- `targets` -- Target values for the new data window.
- `train_size` -- Number of new training samples.
- `blend` -- Blending factor in (0.0, 1.0]. At 0.1 (default), the new model contributes 10% to the updated weights. At 1.0, fully replaces the old model (equivalent to `Train()`).
- `lr`, `epochs`, `weight_decay`, `lr_decay` -- SGD parameters for training the fresh model (same semantics as `Train()`).

**Notes:**
- If no prior `Train()` has been called, delegates to `Train()` (blend is ignored).
- Typical usage: call `Run()` with a new data window, then `TrainIncremental()` to adapt the readout without full retraining.

---

#### Prediction and Evaluation

##### `PredictRaw`

```cpp
[[nodiscard]] float PredictRaw(size_t timestep) const;
```

Returns the raw (continuous) prediction for a single collected timestep.

**Parameters:**
- `timestep` -- Index into collected states, in [0, NumCollected()). Asserts in range.

**Returns:** Continuous float prediction. For classification, threshold at 0.0 to get a class label, or use `Accuracy()` which handles thresholding internally.

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
- `targets` -- Full target array. The slice `targets[start .. start+count)` is used.
- `start` -- First timestep index (indexes into both the feature buffer and the target array).
- `count` -- Number of timesteps to evaluate.

**Returns:** R² value. 1.0 = perfect prediction. 0.0 = predicts the mean. Can be negative (worse than predicting the mean).

**Notes:**
- Asserts `start + count <= NumCollected()`.
- Triggers `EnsureFeatures()` if needed.

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
- `targets` -- Full target array. Slice `targets[start .. start+count)` is used.
- `start` -- First timestep index.
- `count` -- Number of timesteps to evaluate.

**Returns:** NRMSE value. 0.0 = perfect. 1.0 = as bad as predicting the mean. Returns infinity if target variance < 1e-12 (constant signal).

**Notes:**
- Returns 0.0 if `count` is 0.
- The standard metric for reservoir computing benchmarks (Mackey-Glass, NARMA-10).

---

##### `Accuracy`

```cpp
[[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;
```

Computes classification accuracy on a slice of collected states. Predictions are thresholded at 0.0: positive output -> +1, negative -> -1.

**Parameters:**
- `labels` -- Full label array with values {-1.0, +1.0}. Slice `labels[start .. start+count)` is used.
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
| `OutputFraction()` | `float` | The `output_fraction` from config. |
| `OutputStride()` | `size_t` | Stride used for vertex selection: `max(1, N / M)`. |
| `NumOutputVerts()` | `size_t` | Number of selected vertices M = ceil(N / stride). |
| `GetReadoutType()` | `ReadoutType` | Readout type selected at construction. |
| `GetFeatureMode()` | `FeatureMode` | Feature mode selected at construction. |
| `GetAlpha()` | `float` | The tanh gain alpha from the reservoir config. |
| `NumInputs()` | `size_t` | Number of input channels from config. |
| `GetConfig()` | `ReservoirConfig` | Full config used to construct this ESN (for serialization). |

---

##### Readout State Access

The ESN exposes its trained readout state for serialization. The reservoir weights are deterministic from the seed, so only the config and readout state need to be saved.

**`ReadoutState` struct** (nested in `ESN<DIM>`):

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `std::vector<double>` | Weight vector (double for both readout types). |
| `bias` | `double` | Bias term. |
| `feature_mean` | `std::vector<float>` | Per-feature mean from training standardization. |
| `feature_scale` | `std::vector<float>` | Per-feature 1/std from training standardization. |
| `is_trained` | `bool` | True if the readout has been trained. |

| Method | Returns | Description |
|--------|---------|-------------|
| `GetReadoutState()` | `ReadoutState` | Extract trained readout for serialization. |
| `SetReadoutState(state)` | `void` | Restore a previously saved readout state. |

**Example: save and restore a trained model**

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

No external dependencies beyond the C++ standard library.
