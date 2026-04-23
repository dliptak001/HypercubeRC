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
  - [ReadoutConfig](#hcnnreadoutconfig)
  - [CNNTrainHooks](#cnntrainhooks)
  - [ESN\<DIM\>](#esndim)
- [Dependencies](#dependencies)

## What's in the SDK

After installation, the SDK contains:

```
<prefix>/
  include/HypercubeRC/
    ESN.h              -- The public API (the only header consumers include)
    Reservoir.h        -- Internal: included by ESN.h
    Readout.h      -- Transitive: types used by the ESN API (ReadoutConfig, ReadoutTask, CNNTrainHooks)
  lib/
    libHypercubeRCCore.a
  lib/cmake/HypercubeRC/
    HypercubeRCConfig.cmake
    HypercubeRCTargets.cmake
    HypercubeRCConfigVersion.cmake
```

Consumers include `<HypercubeRC/ESN.h>` (installed SDK) or `"ESN.h"` (FetchContent) and link against `HypercubeRC::HypercubeRCCore`. The other headers are present because ESN.h includes them; there is no need to include them directly, but their public types (`ReadoutConfig`, `ReadoutTask`, `CNNTrainHooks`) are part of the API surface.

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
cmake_minimum_required(VERSION 4.1)
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

**Note:** HypercubeCNN must be available at `../HypercubeCNN` relative to the
fetched HypercubeRC source tree (or override `HCNN_DIR` in the CMakeLists).
See [Dependencies](#dependencies).

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

This example uses FetchContent-style includes (`"ESN.h"`). For an installed SDK,
use `<HypercubeRC/ESN.h>` instead.

```cpp
#include "ESN.h"
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
    ESN<DIM> esn(cfg);

    // Drive and train
    esn.Warmup(signal.data(), warmup);
    esn.Run(signal.data() + warmup, collect);

    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t)
        targets[t] = signal[warmup + t + 1];  // predict next value

    size_t train_size = 1400;
    size_t test_size = collect - train_size;

    ReadoutConfig cnn_cfg;
    cnn_cfg.epochs = 25;
    cnn_cfg.batch_size = 128;
    cnn_cfg.lr_max = 0.003f;
    esn.Train(targets.data(), train_size, cnn_cfg);

    double r2 = esn.R2(targets.data(), train_size, test_size);
    std::cout << "R2: " << r2 << "\n";

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
| 13-16 | 8192-65536  | Large-scale research |

### Enums

#### `ReadoutTask`

Task head for the HCNN readout. Declared in `Readout.h`.

| Value | Description |
|-------|-------------|
| `Regression` | MSE loss, de-centered predictions. `num_outputs` sets the number of regression targets. |
| `Classification` | Softmax + cross-entropy loss. `num_outputs` sets the number of classes. Targets are float class indices; predictions are raw logits (use `argmax`). |

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
| `seed` | `uint64_t` | `0` | RNG seed for weight initialization. Every seed (including 0) produces a valid weight topology; different seeds yield measurably different performance. Per-DIM surveyed seeds are in `Presets.h`. |
| `alpha` | `float` | `1.0` | Gain applied inside the tanh activation: `tanh(alpha * sum)`. Values > 1.0 sharpen the nonlinearity; < 1.0 linearize it. |
| `spectral_radius` | `float` | `0.9` | Target spectral norm for recurrent weight matrix. Controls the echo state property -- how quickly past inputs fade. Scale-invariant across all DIM values (vertex-transitive topology property). |
| `leak_rate` | `float` | `1.0` | Leaky integrator coefficient. `state = (1 - leak_rate) * old_state + leak_rate * activation`. At 1.0 (default), each step fully replaces state. Values < 1.0 add temporal smoothing. |
| `input_scaling` | `float` | `0.02` | Magnitude of input weights, drawn from U(-input_scaling, +input_scaling). Scale-invariant across all DIM values. |
| `num_inputs` | `size_t` | `1` | Number of input channels. In multi-input mode (K channels), channel k drives every K-th vertex starting at offset k (stride-interleaved). |
| `output_fraction` | `float` | `1.0` | Fraction of N vertices used as readout features, in range (0.0, 1.0]. Must yield a power-of-2 stride. At 0.5, a stride-selected sub-hypercube of N/2 vertices is passed to the readout. |

---

### ReadoutConfig

Configuration struct for the HCNN readout's architecture and training.

```cpp
struct ReadoutConfig {
    int num_outputs      = 1;
    ReadoutTask task        = ReadoutTask::Regression;
    int num_layers       = 0;        // 0 = auto: min(DIM-2, 2)
    int conv_channels    = 16;       // doubles per layer
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.005f;
    float lr_min_frac    = 0.1f;
    int   lr_decay_epochs = 0;       // 0 = use epochs
    float weight_decay   = 0.0f;
    unsigned seed        = 42;
    bool verbose         = false;
    bool verbose_train_acc = false;
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_outputs` | `int` | `1` | Number of output neurons. For regression: number of targets. For classification: number of classes. |
| `task` | `ReadoutTask` | `Regression` | Task head. See [ReadoutTask](#hcnntask). |
| `num_layers` | `int` | `0` (auto) | Number of Conv+Pool pairs. `0` auto-computes `min(DIM - 2, 2)`. Each Pool halves the hypercube dimension, capped by `DIM - 2` (HCNNConv requires DIM >= 3). |
| `conv_channels` | `int` | `16` | Base channel count for the first Conv layer. Doubles per layer (16, 32 for a 2-layer stack). |
| `epochs` | `int` | `200` | Training epochs. Structured signals saturate at ~25 epochs; chaotic signals (NARMA) need ~2000. |
| `batch_size` | `int` | `32` | Mini-batch size. Use 128 on multi-core CPUs to saturate threading. |
| `lr_max` | `float` | `0.005` | Peak learning rate for cosine annealing. **Keep <= 0.005 to avoid weight divergence into denormal/NaN territory.** |
| `lr_min_frac` | `float` | `0.1` | Cosine schedule floor as fraction of `lr_max`. Effective `lr_min = lr_max * lr_min_frac`. |
| `lr_decay_epochs` | `int` | `0` | Cosine decay horizon. 0 = use `epochs`. Set > epochs to trace only a prefix of the cosine curve (keeps lr high when shortening a run). |
| `weight_decay` | `float` | `0.0` | L2 weight decay applied by the Adam optimizer. |
| `seed` | `unsigned` | `42` | Seed for weight initialization. |
| `verbose` | `bool` | `false` | Print per-epoch lr to stdout. |
| `verbose_train_acc` | `bool` | `false` | Also compute and print training accuracy (classification) or MSE (regression) each epoch. Costs one extra forward pass per epoch. |

**Architecture auto-sizing:** For all supported DIMs (5-16), auto-sizing produces 2 Conv+Pool layers with channels 16, 32 and a final hypercube dimension of DIM - 2. Override `num_layers` to change this. See `docs/Readout.md` for the full design notes.

---

### CNNTrainHooks

Runtime-only training hooks for mid-training evaluation. Kept separate from
`ReadoutConfig` because the config must stay POD for checkpoint serialization.

```cpp
struct CNNTrainHooks {
    int eval_every_epochs = 0;
    std::function<void(int epoch_done, int total_epochs, float lr)>
        epoch_callback;
    bool stop_requested = false;
};
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eval_every_epochs` | `int` | `0` | Callback interval. 0 disables mid-training callbacks. |
| `epoch_callback` | `std::function` | empty | Fires after every `eval_every_epochs` completed epochs, and unconditionally after the final epoch. During the callback, the readout is usable for Predict/R2/Accuracy. |
| `stop_requested` | `bool` | `false` | Set to `true` inside the callback to end training early. |

---

### ESN\<DIM\>

The complete pipeline wrapper: Reservoir -> Readout.

```cpp
// Construction
ESN<DIM>(cfg);

// Reservoir driving
esn.Warmup(inputs, num_steps);
esn.Run(inputs, num_steps);
esn.ClearStates();
esn.ResetReservoirOnly();
esn.SaveReservoirState(state_out, output_out);
esn.RestoreReservoirState(state_in, output_in);

// Batch training
esn.Train(targets, train_size);
esn.Train(targets, train_size, cnn_cfg);
esn.Train(targets, train_size, cnn_cfg, hooks);

// Online (streaming) training
esn.InitOnline(warmup_inputs, warmup_count, cnn_cfg);
esn.TrainLiveStep(target_class, lr, weight_decay);
esn.TrainLiveBatch(states, targets, count, lr, weight_decay);
esn.TrainLiveStepRegression(target, lr, weight_decay);
esn.TrainLiveBatchRegression(states, targets, count, lr, weight_decay);
esn.ComputeTargetCentering(targets, num_samples);
esn.CopyLiveState(out);

// Prediction & evaluation (collected states)
esn.PredictRaw(timestep);
esn.PredictRaw(timestep, output);
esn.R2(targets, start, count);
esn.NRMSE(targets, start, count);
esn.Accuracy(labels, start, count);

// Prediction (live reservoir state)
esn.PredictLiveRaw();
esn.PredictLiveRaw(output);

// State access
esn.SelectedStates();
esn.NumCollected();
esn.NumOutputs();
esn.NumOutputVerts();
esn.OutputFraction();
esn.NumInputs();
esn.GetConfig();
esn.GetReadoutState();
esn.SetReadoutState(state);
esn.SetCNNConfig(cfg);
```

---

#### Construction

```cpp
explicit ESN(const ReservoirConfig& cfg);
```

Creates the reservoir from `cfg`. Reservoir weights are generated and spectral-radius-rescaled at construction time. The HCNN readout is initialized lazily when `Train()` or `InitOnline()` is called.

**Parameters:**
- `cfg` -- Reservoir configuration. See [ReservoirConfig](#reservoirconfig).

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

---

##### `Run`

```cpp
void Run(const float* inputs, size_t num_steps);
```

Drives the reservoir for `num_steps` timesteps, recording the full N-dimensional state vector at each step. States are appended to the internal buffer -- multiple `Run()` calls accumulate.

**Parameters:**
- `inputs` -- Pointer to `num_steps * num_inputs` floats, row-major. Same layout as `Warmup()`.
- `num_steps` -- Number of timesteps to drive and record.

---

##### `ClearStates`

```cpp
void ClearStates();
```

Clears all collected states. The reservoir's live internal state is **not** reset -- it retains its current activation. The trained readout is also preserved.

Use this between independent sequences: clear the collected data, then `Warmup()` + `Run()` on a new input sequence without rebuilding the ESN.

---

##### `ResetReservoirOnly`

```cpp
void ResetReservoirOnly();
```

Zeros the reservoir's internal state (both state and output buffers). Recurrent weights, input weights, and all hyperparameters are untouched. Collected states are **not** cleared. The trained readout is preserved.

Use for episodic tasks where each episode starts from a clean slate (e.g., per-expression reset in character-level sequence tasks).

---

##### `SaveReservoirState` / `RestoreReservoirState`

```cpp
void SaveReservoirState(float* state_out, float* output_out) const;
void RestoreReservoirState(const float* state_in, const float* output_in);
```

Snapshot and restore the reservoir's live internal state. Each buffer must hold N floats. Use for mid-training evaluation: save state, run an eval pass, then restore to resume training from the same point.

---

#### Batch Training

##### `Train` (default parameters)

```cpp
void Train(const float* targets, size_t train_size);
```

Trains the HCNN readout on the first `train_size` collected states using default-constructed `ReadoutConfig`.

**Parameters:**
- `targets` -- Target values. Layout depends on task (see HCNN overload below).
- `train_size` -- Number of samples to train on, starting from collected state index 0.

---

##### `Train` (with ReadoutConfig)

```cpp
void Train(const float* targets, size_t train_size,
           const ReadoutConfig& config);
```

Trains the HCNN readout with explicit hyperparameters.

**Parameters:**
- `targets` -- Target values. Layout depends on `config.task`:
  - **Regression:** `train_size * config.num_outputs` floats, row-major.
  - **Classification:** `train_size` floats (float class indices).
- `train_size` -- Number of training samples from collected state index 0.
- `config` -- Architecture and training hyperparameters. See [ReadoutConfig](#hcnnreadoutconfig).

---

##### `Train` (with hooks)

```cpp
void Train(const float* targets, size_t train_size,
           const ReadoutConfig& config,
           CNNTrainHooks& hooks);
```

Same as above but fires `hooks.epoch_callback` at the specified interval. During the callback the readout is usable for evaluation. Set `hooks.stop_requested = true` to end training early.

---

#### Online (Streaming) Training

For applications where data arrives continuously. The reservoir advances one step at a time; the readout is updated via per-sample or mini-batch gradient steps.

##### `InitOnline`

```cpp
void InitOnline(const float* warmup_inputs, size_t warmup_count,
                const ReadoutConfig& config);
```

Initializes the HCNN readout for online training. Internally calls `Run()` (not `Warmup()`) on the warmup inputs, computes per-vertex standardization statistics from the resulting states, builds the CNN architecture, sets the Adam optimizer, then clears collected states. After this call the reservoir's live state reflects having processed `warmup_count` steps, but `NumCollected()` is 0. Call before any `TrainLive*` method.

**Parameters:**
- `warmup_inputs` -- Warmup signal: `warmup_count * num_inputs` floats. These drive the reservoir forward (not discarded like `Warmup()`).
- `warmup_count` -- Number of warmup timesteps. Must be large enough for representative standardization statistics.
- `config` -- HCNN architecture config. `config.epochs` is unused in online mode.

---

##### `TrainLiveStep` (classification)

```cpp
void TrainLiveStep(float target_class, float lr, float weight_decay = 0.0f);
```

Single-sample online gradient step on the reservoir's current live state. Classification only -- `target_class` is cast to int internally.

---

##### `TrainLiveBatch` (classification)

```cpp
void TrainLiveBatch(const float* states, const int* targets,
                    size_t count, float lr, float weight_decay = 0.0f);
```

Mini-batch online gradient step. `states` is `count` rows of `NumOutputVerts()` floats (from `CopyLiveState`). `targets` is `count` int class indices. Parallelized across threads.

---

##### `TrainLiveStepRegression`

```cpp
void TrainLiveStepRegression(const float* target, float lr,
                             float weight_decay = 0.0f);
```

Single-sample online gradient step on the reservoir's current live state. `target` is `NumOutputs()` floats.

---

##### `TrainLiveBatchRegression`

```cpp
void TrainLiveBatchRegression(const float* states, const float* targets,
                              size_t count, float lr, float weight_decay = 0.0f);
```

Mini-batch online gradient step for regression. `states` is `count` rows of `NumOutputVerts()` floats. `targets` is `count * NumOutputs()` floats (row-major).

---

##### `ComputeTargetCentering`

```cpp
void ComputeTargetCentering(const float* targets, size_t num_samples);
```

Computes per-output target means from sample targets and stores them. Call after `InitOnline` for regression tasks so that online training subtracts the mean internally and `PredictRaw` / `PredictLiveRaw` add it back (matching batch training behavior).

---

##### `CopyLiveState`

```cpp
void CopyLiveState(float* out) const;
```

Copies the current subsampled reservoir state into `out` (`NumOutputVerts()` floats). Use to accumulate states for `TrainLiveBatch` / `TrainLiveBatchRegression`.

---

#### Prediction and Evaluation (Collected States)

##### `PredictRaw` (scalar)

```cpp
[[nodiscard]] float PredictRaw(size_t timestep) const;
```

Returns the scalar prediction for a collected timestep. Asserts `NumOutputs() == 1`.

**Parameters:**
- `timestep` -- Index into collected states, in [0, NumCollected()).

---

##### `PredictRaw` (multi-output)

```cpp
void PredictRaw(size_t timestep, float* output) const;
```

Writes `NumOutputs()` floats to `output` for a collected timestep. For regression: de-centered predictions. For classification: raw logits (apply argmax for predicted class).

---

##### `R2`

```cpp
[[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;
```

R-squared on collected timesteps [start, start+count).

**Parameters:**
- `targets` -- Must span timesteps [0, start+count): `(start+count) * NumOutputs()` floats (row-major). Indexed from `targets[start * NumOutputs()]`.
- `start` -- First timestep index.
- `count` -- Number of timesteps to evaluate.

**Returns:** R² averaged across outputs. 1.0 = perfect. Can be negative.

---

##### `NRMSE`

```cpp
[[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;
```

Normalized RMSE on collected timesteps. Same `targets` layout as `R2`.

**Returns:** NRMSE averaged across outputs. 0.0 = perfect. 1.0 = predicts the mean.

---

##### `Accuracy`

```cpp
[[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;
```

Classification accuracy on collected timesteps.

**Parameters:**
- `labels` -- Must span timesteps [0, start+count): `(start+count)` floats (class indices). Indexed from `labels[start]`.
- `start` -- First timestep index.
- `count` -- Number of timesteps to evaluate.

**Returns:** Fraction correct in [0.0, 1.0].

---

#### Prediction (Live Reservoir State)

For streaming inference without collecting states.

##### `PredictLiveRaw` (scalar)

```cpp
[[nodiscard]] float PredictLiveRaw() const;
```

Scalar prediction from the reservoir's current live state. Asserts `NumOutputs() == 1`.

---

##### `PredictLiveRaw` (multi-output)

```cpp
void PredictLiveRaw(float* output) const;
```

Writes `NumOutputs()` floats to `output` from the reservoir's current live state. For autoregressive / streaming inference loops.

---

#### State Access and Persistence

##### `SelectedStates`

```cpp
[[nodiscard]] std::vector<float> SelectedStates() const;
```

Returns stride-selected vertices from all collected states: `NumCollected() * NumOutputVerts()` floats, row-major.

---

##### Accessors

| Method | Returns | Description |
|--------|---------|-------------|
| `NumCollected()` | `size_t` | Timesteps recorded by `Run()`. |
| `NumOutputs()` | `size_t` | From `ReadoutConfig::num_outputs` after training. |
| `OutputFraction()` | `float` | The `output_fraction` from config. |
| `NumOutputVerts()` | `size_t` | Number of selected vertices M = ceil(N / stride). |
| `NumInputs()` | `size_t` | Number of input channels from config. |
| `GetConfig()` | `ReservoirConfig` | Full config used to construct this ESN. |

---

##### `SetCNNConfig`

```cpp
void SetCNNConfig(const ReadoutConfig& cfg);
```

Pre-set the HCNN architecture config on the readout. Required before `SetReadoutState` when restoring a saved model without training -- the readout needs the config to reconstruct the CNN architecture before injecting weights.

---

##### Readout State Serialization

The ESN exposes its trained readout state for save/restore. The reservoir weights are deterministic from the seed, so only the config and readout state need to be persisted.

**`ReadoutState` struct** (nested in `ESN<DIM>`):

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `std::vector<double>` | Opaque flattened blob of all conv kernels, biases, and dense-head weights. Round-trip only -- do not interpret. |
| `bias` | `double` | Fallback for per-output target centering (backward compat with old checkpoints). |
| `feature_mean` | `std::vector<float>` | Per-vertex mean for the selected output vertices (`NumOutputVerts()` entries). |
| `feature_scale` | `std::vector<float>` | Per-vertex 1/std for the selected output vertices. |
| `target_mean` | `std::vector<double>` | Per-output target centering (regression). Empty for classification. |
| `is_trained` | `bool` | True if the readout has been trained. |

| Method | Description |
|--------|-------------|
| `GetReadoutState()` | Extract trained readout for serialization. |
| `SetReadoutState(state)` | Restore a previously saved readout state. Reconstructs the CNN from stored config + weight blob. |

**Example: save and restore a trained model**

```cpp
// Save
auto cfg = esn.GetConfig();
auto cnn_cfg = ...; // the ReadoutConfig used for training
auto state = esn.GetReadoutState();
// serialize cfg, cnn_cfg, state using your preferred format

// Restore
ESN<7> restored(cfg);
restored.SetCNNConfig(cnn_cfg);
restored.SetReadoutState(state);
// Ready to predict -- no retraining needed.
```

---

## Dependencies

HypercubeRC depends on a single external project:

**HypercubeCNN** -- sibling library providing the hypercube convolutional network used by `Readout`.

- Expected location: `../HypercubeCNN` relative to the HypercubeRC source tree (adjust `HCNN_DIR` in `CMakeLists.txt` if yours is elsewhere).
- Built as `libHypercubeCNNCore.a` inside its own `cmake-build-release` directory before HypercubeRC is configured.
- Public headers are re-exported through `HypercubeRCCore`'s include interface, so consumers of HypercubeRC do not need to add HypercubeCNN to their own link line -- `target_link_libraries(my_app PRIVATE HypercubeRCCore)` pulls it in transitively.
- The HCNN headers used by HypercubeRC consumers are the ones re-exported by `Readout.h` (forward-declared `hcnn::HCNN` via PIMPL); the full HCNN API is not part of the public HypercubeRC surface.

No other external dependencies beyond the C++ standard library.
