# HCNNReadout — HypercubeCNN as ESN Readout Layer

## Overview

HCNNReadout integrates HypercubeCNN (HCNN) as a learned readout for the
HypercubeRC reservoir computing pipeline.  Instead of linear regression on
hand-crafted features, the CNN's convolution kernels discover which vertex
interactions predict the target directly from raw reservoir state.

**Data path:** raw state (N = 2^DIM) -> input standardization ->
HCNN (Conv->Pool stack -> GAP -> Linear) -> de-center -> output.

## Supported DIM Range

| Property         | Value    | Source                           |
|------------------|----------|----------------------------------|
| HypercubeCNN     | 3-32     | `HCNNNetwork.cpp:24`             |
| HypercubeRC ESN  | 5-16     | `ESN.cpp` template instantiations |
| HCNNReadout       | **5-16** | Intersection; matches ESN range  |

## Architecture Auto-sizing

Each Conv+Pool pair reduces DIM by 1.  HCNNConv requires DIM >= 3, so the
maximum pairs for a given start DIM is `DIM - 3`.

Default behavior (`num_layers = 0`): auto-compute `min(DIM - 3, 4)`.
Channels double per layer from `conv_channels` base (16 -> 32 -> 64 -> 128).

| DIM  | auto layers | channels             | final DIM |
|------|-------------|----------------------|-----------|
| 5    | 1           | 16                   | 4         |
| 6    | 2           | 16, 32               | 4         |
| 7    | 3           | 16, 32, 64           | 4         |
| 8    | 4 (cap)     | 16, 32, 64, 128      | 4         |
| 9    | 4 (cap)     | 16, 32, 64, 128      | 5         |
| 10   | 4 (cap)     | 16, 32, 64, 128      | 6         |
| 11   | 4 (cap)     | 16, 32, 64, 128      | 7         |
| 12   | 4 (cap)     | 16, 32, 64, 128      | 8         |
| 13   | 4 (cap)     | 16, 32, 64, 128      | 9         |
| 14   | 4 (cap)     | 16, 32, 64, 128      | 10        |
| 15   | 4 (cap)     | 16, 32, 64, 128      | 11        |
| 16   | 4 (cap)     | 16, 32, 64, 128      | 12        |

Manual override: set `config.num_layers` to a specific count (asserted <= DIM - 3).

## Task Types

| Task             | targets layout                         | Output             | Metric   |
|------------------|----------------------------------------|--------------------|----------|
| Regression       | num_samples x num_outputs (row-major)  | de-centered floats | R2, NRMSE |
| Classification   | num_samples floats (class indices)     | logits (argmax)    | Accuracy |

Configured via `HCNNReadoutConfig::task` (`HCNNTask::Regression` / `Classification`)
and `HCNNReadoutConfig::num_outputs`.

## HCNNReadoutConfig

```cpp
struct HCNNReadoutConfig {
    int num_outputs   = 1;        // classes or regression targets
    HCNNTask task     = HCNNTask::Regression;
    int num_layers    = 0;        // 0 = auto: min(DIM-3, 4)
    int conv_channels = 16;       // base channels (doubles per layer)
    int epochs        = 200;
    int batch_size    = 32;
    float lr_max      = 0.005f;   // cosine annealing peak
    float lr_min_frac = 0.1f;     // floor = lr_max * lr_min_frac
    float weight_decay = 0.0f;
    unsigned seed     = 42;
};
```

## Serialization

Weights are flattened/restored via `HCNN::GetWeights()` / `SetWeights()`.
Layout: conv kernels + biases (per layer, in order) then readout weights + bias.
Input standardization (mean/scale) and target centering are stored separately
via the ESN `ReadoutState` struct.

`rebuild_from_blob()` reconstructs the full HCNN network from stored `config_`
and `dim_` via `build_architecture()`, then injects the weight blob -- no
retraining needed.

## ESN Integration Points

- \`ReadoutType::HCNN\` selects this readout
- `ESN::Train(targets, train_size)` routes to `HCNNReadout::Train` with default config
- `ESN::Train(targets, train_size, HCNNReadoutConfig)` for custom config
- `ESN::PredictRaw(timestep)` scalar (num_outputs must be 1)
- `ESN::PredictRaw(timestep, float* output)` multi-output
- `ESN::NumOutputs()` returns 1 for Linear/Ridge, config-based for HCNN
- `ESN::R2/NRMSE/Accuracy` handle multi-output target layout for HCNN

## Integration Status

| Feature                          | Status    |
|----------------------------------|-----------|
| Single-output regression         | Done      |
| Multi-output regression          | Done      |
| Multi-class classification       | Done      |
| Weight serialization / restore   | Done      |
| Compile-time visitor guard       | Done      |
| FeatureMode::Raw enforcement     | Done      |
| Auto-sized architecture from DIM | Done      |
| DIM 5-16 support                 | Done      |
| Incremental/streaming training   | Won't fix (by design -- use HCNN online training for streaming) |
