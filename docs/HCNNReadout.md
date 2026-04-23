# HCNNReadout — HypercubeCNN as ESN Readout Layer

## HypercubeCNN

[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) is a standalone
convolutional neural network library that replaces the familiar 2D pixel
grid with a DIM-dimensional Boolean hypercube (N = 2^DIM vertices).
Convolution kernels are defined by Hamming-distance neighborhoods — each
vertex has exactly DIM neighbors, reached by flipping one bit — making
weight sharing mathematically exact under the hypercube's Z₂ⁿ symmetry
group.  There are no adjacency lists, no padding, and no border effects;
neighbor lookup is a single XOR instruction.

Pooling pairs each vertex with its bitwise complement (the maximally
distant point) and reduces DIM by 1, producing a perfect sub-hypercube.
Stacking Conv+Pool stages builds a feature hierarchy analogous to spatial
CNNs, with DIM shrinking and channel count growing at each stage.  The
library supports both classification (softmax + cross-entropy) and
regression (MSE) through a unified pipeline, trained end-to-end with
backpropagation and Adam.

### Why HypercubeCNN as the readout for HypercubeRC?

The pairing is topology-native.  HypercubeRC's reservoir is a Boolean
hypercube: N = 2^DIM neurons sit on hypercube vertices, connected by
XOR-addressed edges.  The reservoir state *is* a signal on a hypercube
graph.  HypercubeCNN's convolutions are built to operate on exactly that
structure — Hamming-distance kernels that respect the same vertex
addressing and neighbor relationships the reservoir uses internally.

This means the readout consumes the reservoir's state with zero
topological distortion.  There is no reshaping into a flat vector for a
linear fit, no arbitrary packing into a 2D grid for a spatial CNN — the
data stays on the hypercube it was born on, and the convolution kernels
exploit the same adjacency structure that generated the dynamics.
Locality on the reservoir graph maps directly to locality in the
convolution kernel: neighbors that influenced each other during reservoir
evolution are neighbors in the learned feature extraction.

A conventional readout (ridge regression on the flattened state vector)
treats the N activations as unstructured features and ignores the graph
entirely.  A spatial CNN would impose a fabricated 2D geometry that has
no relationship to the reservoir's actual connectivity.  HypercubeCNN is
the only architecture where the readout's inductive bias matches the
reservoir's topology — making it the natural learned readout for a
hypercube reservoir.

## Role in the Pipeline

```
Reservoir (N states) ────────────────────────────────> HCNNReadout
    fixed random                                          TRAINED
```

The readout is the only trained component in the entire pipeline. The
reservoir's random weights are fixed at initialization. All learning
happens here.

This is the core principle of reservoir computing: a complex, nonlinear
dynamical system (the reservoir) projects inputs into a high-dimensional
space, and the readout learns to extract the desired output from that
projection. HCNNReadout replaces a linear fit with a learned
convolutional network that operates directly on raw reservoir state,
discovering its own nonlinear features.

**Data path:** raw state (N = 2^DIM) -> input standardization ->
HCNN (Conv->Pool stack -> Flatten -> Linear) -> de-center -> output.

## Supported DIM Range

| Property         | Value    | Source                           |
|------------------|----------|----------------------------------|
| HypercubeCNN     | 3-32     | `HCNNNetwork.cpp:24`             |
| HypercubeRC ESN  | 5-16     | `ESN.cpp` template instantiations |
| HCNNReadout      | **5-16** | Intersection; matches ESN range  |

## Architecture Auto-sizing

Each Conv+Pool pair reduces DIM by 1. HCNNConv requires DIM >= 3, so
the maximum pairs for a given start DIM is `DIM - 2`.

Default behavior (`num_layers = 0`): auto-compute `min(DIM - 2, 2)`.
For all supported DIMs (5-16), the cap of 2 is always hit, producing a
2-layer Conv+Pool stack with channels doubling per layer (16 → 32 at
default `conv_channels`). Final DIM after pooling = DIM - 2.

Manual override: set `config.num_layers` to a specific count (asserted
<= DIM - 2). The auto-computed value is also clamped to at least 1.

## Training Algorithm

A stack of hypercube Conv+MaxPool layers followed by flatten and a
dense head, trained with Adam and cosine-annealed learning rate.

1. Standardize the raw N-vertex state (per-vertex mean/std from the
   training set).
2. Auto-size the architecture from DIM: `min(DIM - 2, 2)` Conv+Pool
   pairs, base `conv_channels` doubling per layer (16 -> 32).
   Each Pool halves the hypercube dimension, so the stack depth is
   capped by `DIM - 2` (HCNNConv requires DIM >= 3).
3. For each epoch, shuffle samples into mini-batches and run Adam
   forward/backward over the full training set. Learning rate follows
   a cosine schedule from `lr_max` down to `lr_max * lr_min_frac`.
4. Two task heads:
   - **Regression** -- per-output target centering, MSE loss, de-centered
     predictions at inference.
   - **Classification** -- integer class labels, softmax+cross-entropy,
     logit output via `PredictRaw` or argmax via `PredictClass`.
5. After training, weights are flattened via `HCNN::GetWeights()` for
   serialization and restored via `SetWeights()` on reload.

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
    int num_outputs      = 1;        // classes or regression targets
    HCNNTask task        = HCNNTask::Regression;
    int num_layers       = 0;        // 0 = auto: min(DIM-2, 2)
    int conv_channels    = 16;       // base channels (doubles per layer)
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.005f;   // cosine annealing peak
    float lr_min_frac    = 0.1f;     // floor = lr_max * lr_min_frac
    int   lr_decay_epochs = 0;       // cosine decay horizon; 0 = use epochs
    float weight_decay   = 0.0f;
    unsigned seed        = 42;
    bool verbose         = false;    // print per-epoch lr
    bool verbose_train_acc = false;  // also print train accuracy/MSE each epoch
};
```

**Cost:** O(epochs * samples * layer_flops). For a typical DIM=8
configuration (~256 states per sample, 2 Conv+Pool pairs, ~20k samples,
300 epochs) this runs in seconds to minutes depending on core count.
CPU cores saturate at `batch_size >= 128`.

**Stability note:** `lr_max` above ~0.005 can drive weights into
denormal/NaN territory, where CPU falls off fast math paths and
throughput collapses.

## Feature Standardization

Raw N-vertex state is centered and scaled per-vertex (zero mean, unit
variance) before it enters the first Conv layer. Statistics are computed
from the training set and stored; `PredictRaw` applies the same
transform at inference. The convolution kernels discover their own
nonlinear features from raw state.

## When to Use

- Tasks where a linear-readout ceiling is hit and nonlinear feature
  discovery is worth the training cost.
- Classification problems. HCNN natively supports multi-class with
  softmax+cross-entropy. See `examples/SignalClassification.cpp`.
- DIM 7+ where the auto-sized architecture gets enough Conv+Pool depth
  to be expressive.

**When not to use:** Small-DIM tasks (5-6) where the architecture has
minimal depth and training cost isn't justified by accuracy gains.

## Online Training API

HCNNReadout supports per-sample and mini-batch online gradient steps
for streaming applications where data arrives continuously.

### Setup

`InitOnline(warmup_states, warmup_count, dim, config)` computes
per-vertex standardization statistics from a warmup buffer, builds
the architecture, and sets the Adam optimizer. No batch `Train()` call
is involved -- the network starts from random weights and learns
entirely from online updates.

For regression tasks, call `ComputeTargetCentering(targets, n)` after
`InitOnline` to store per-output target means. Online steps then
subtract the mean internally and `PredictRaw` adds it back, matching
batch behavior.

### Gradient steps

| Method | Task | Granularity |
|--------|------|-------------|
| `TrainOnlineStep(state, class, lr, wd)` | Classification | Single sample |
| `TrainOnlineBatch(states, classes, count, lr, wd)` | Classification | Mini-batch |
| `TrainOnlineStepRegression(state, target, lr, wd)` | Regression | Single sample |
| `TrainOnlineBatchRegression(states, targets, count, lr, wd)` | Regression | Mini-batch |

All methods standardize inputs internally using the statistics from
`InitOnline`. Regression targets are centered internally if
`ComputeTargetCentering` was called. Mini-batch variants are
parallelized via `HCNN::TrainBatch`.

See `examples/HRCCNN_LM_Text/` for a working streaming implementation
and `examples/StreamingAnomaly.cpp` for an anomaly-detection use case.

## Serialization

Weights are flattened/restored via `HCNN::GetWeights()` / `SetWeights()`.
Layout: conv kernels + biases (per layer, in order) then readout weights + bias.
Input standardization (mean/scale) and target centering are stored separately
via the ESN `ReadoutState` struct.

`rebuild_from_blob()` reconstructs the full HCNN network from stored `config_`
and `dim_` via `build_architecture()`, then injects the weight blob -- no
retraining needed.

## HCNNReadout Public Interface

`HCNNReadout` is the readout class used by `ESN<DIM>`. ESN holds it as a
direct `HCNNReadout readout_` member and delegates training, prediction,
and evaluation to it. The methods below are on HCNNReadout; see
`docs/CPP_SDK.md` for the ESN-level wrappers.

| Method | Returns |
|--------|---------|
| `Train(states, targets, num_samples, dim, config)` | void |
| `Train(states, targets, num_samples, dim, config, hooks)` | void |
| `PredictRaw(state)` | float (scalar, single-output) |
| `PredictRaw(state, output)` | void (multi-output) |
| `PredictClass(state)` | int (argmax over logits) |
| `R2(states, targets, num_samples)` | double |
| `Accuracy(states, labels, num_samples)` | double |
| `Weights()` | `vector<double>` (flattened blob) |
| `Bias()` | double (target mean fallback) |
| `NumFeatures()` | size_t (= N for raw state) |
| `NumOutputs()` | size_t |

### ESN Integration Points

- `ESN::Train(targets, train_size)` → `HCNNReadout::Train` with default config
- `ESN::Train(targets, train_size, config)` → custom config
- `ESN::Train(targets, train_size, config, hooks)` → with mid-training callbacks
- `ESN::PredictRaw(timestep)` → scalar (asserts `num_outputs == 1`)
- `ESN::PredictRaw(timestep, float* output)` → multi-output
- `ESN::NumOutputs()` → delegates to `HCNNReadout::NumOutputs()`
- `ESN::R2/NRMSE/Accuracy` → handle multi-output target layout and state subsampling

## Implementation Notes

- Lives in `readout/` with separate .h/.cpp files.
- Holds a `std::unique_ptr<hcnn::HCNN>` via PIMPL so that
  `#include "HCNN.h"` stays in the .cpp only.
- Not templated -- accepts arbitrary feature counts at runtime.
- Does not store training data -- only learned weights,
  standardization statistics, and the config used to rebuild
  the network on reload.
