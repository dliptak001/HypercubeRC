# HypercubeRC SDK

Static C++ library for reservoir computing on Boolean hypercube graphs.

## What's in the SDK

After installation, the SDK contains:

```
<prefix>/
  include/HypercubeRC/
    ESN.h              -- Main API: pipeline wrapper (Reservoir + Translation + Readout)
    Reservoir.h        -- Hypercube reservoir (N = 2^DIM neurons)
    TranslationLayer.h -- Feature expansion: M states -> 2.5M features
    LinearReadout.h    -- Online SGD readout with L2 decay
    RidgeRegression.h  -- Closed-form (X'X + lambda*I)^-1 X'y readout
  lib/
    libHypercubeRCCore.a
  lib/cmake/HypercubeRC/
    HypercubeRCConfig.cmake
    HypercubeRCTargets.cmake
    HypercubeRCConfigVersion.cmake
```

## Building from source

Requirements: C++23 compiler with OpenMP support (GCC 13+, Clang 17+, MSVC 2022+), CMake 4.1+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/sdk
```

## Using the SDK

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
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

## API overview

### ESN (main entry point)

```cpp
// Construction
ESN<DIM>(cfg, ReadoutType::Ridge, FeatureMode::Translated);  // all params have defaults

// Reservoir driving
esn.Warmup(inputs, num_steps);          // drive without recording
esn.Run(inputs, num_steps);             // drive and collect states
esn.ClearStates();                      // reset (preserves trained readout)

// Training
esn.Train(targets, train_size);                         // default parameters
esn.Train(targets, train_size, lambda);                 // Ridge: custom lambda
esn.Train(targets, train_size, lr, epochs);             // Linear: custom SGD
esn.TrainIncremental(targets, train_size, blend);       // streaming (Linear only)

// Evaluation -- start indexes into both features and targets
esn.PredictRaw(timestep);                // single prediction
esn.R2(targets, start, count);           // R-squared
esn.NRMSE(targets, start, count);        // normalized RMSE
esn.Accuracy(labels, start, count);      // classification accuracy

// State access
esn.Features();                          // cached feature buffer
esn.States();                            // raw N-dimensional state buffer
esn.NumFeatures();                       // M (Raw) or 2.5M (Translated)
esn.EnsureFeatures();                    // force feature computation
```

### ReservoirConfig

```cpp
ReservoirConfig cfg;
cfg.seed            = 42;       // RNG seed for weight initialization
cfg.alpha           = 1.0f;     // tanh gain
cfg.spectral_radius = 0.9f;     // recurrent weight scaling (scale-invariant default)
cfg.leak_rate       = 1.0f;     // 1.0 = full replacement, <1.0 = leaky integrator
cfg.input_scaling   = 0.02f;    // input weight magnitude (scale-invariant default)
cfg.num_inputs      = 1;        // number of input channels
cfg.output_fraction = 1.0f;     // fraction of neurons used as readout features
```

### Template parameter

`DIM` is a compile-time template parameter (5-12). The reservoir has N = 2^DIM neurons:

| DIM | Neurons | Typical use |
|-----|---------|-------------|
| 5   | 32      | Fast prototyping, embedded |
| 6   | 64      | Light benchmarks |
| 7   | 128     | Standard benchmarks |
| 8   | 256     | Production, complex tasks |
| 9-12 | 512-4096 | Research, high-capacity tasks |

For DIM 9+, reduce `output_fraction` to control Ridge readout cost (e.g., 0.25 for DIM 10).

### Feature modes

- **`FeatureMode::Translated`** (default) -- Expands M selected states into 2.5M features via [x | x^2 | x*x_antipodal]. The squared and antipodal-product terms help a linear readout decode information that tanh compresses. Reduces NRMSE by 20-70% on standard benchmarks.

- **`FeatureMode::Raw`** -- Uses M selected states directly. Fewer features, faster computation, sufficient for simple tasks.

### Readout types

- **`ReadoutType::Ridge`** (default) -- Closed-form solution. Fast, deterministic, works well with translation features. Tune via lambda parameter.

- **`ReadoutType::Linear`** -- Online SGD with L2 decay and pocket selection. Supports incremental/streaming training via `TrainIncremental()`.

## Dependencies

The library uses OpenMP internally for parallelization. Consumers do not need OpenMP to compile against the headers -- the OpenMP runtime is linked automatically via CMake's target dependency system.

No other external dependencies. Standard C++ library only.
