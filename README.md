# HypercubeRC

A reservoir computer whose N neurons are arranged on a Boolean hypercube -- a DIM-dimensional graph where each vertex is addressed by a DIM-bit binary index, and two vertices are neighbors if their indices differ by one bit. Input and output are continuous scalar values.

Licensed under the [Apache License 2.0](LICENSE).

## What is Reservoir Computing?

Reservoir computing is a machine learning paradigm for temporal data. Instead of training
a recurrent neural network end-to-end (expensive, unstable, prone to vanishing gradients),
reservoir computing splits the problem in two:

1. **A fixed, random recurrent network (the reservoir)** receives input and produces a
   high-dimensional, time-dependent state. Its weights are never trained — they are set
   once at initialization and frozen.

2. **A simple linear readout** learns to map the reservoir's state to the desired output.
   This is a single-layer linear regression — fast, stable, and globally optimal.

The insight is that a sufficiently rich dynamical system, driven by input, will naturally
create a high-dimensional embedding of the input history. The nonlinear recurrence
provides the computational power; the linear readout provides the learning. This
separation makes reservoir computing orders of magnitude faster to train than
backpropagation-based RNNs, while achieving competitive performance on tasks that
require memory and nonlinear computation.

## What is HypercubeRC?

HypercubeRC is a specific reservoir computing implementation where the reservoir's
connectivity is defined by a Boolean hypercube graph of dimension DIM, giving N = 2^DIM
vertices (neurons). Each vertex's neighbors are computed by XOR operations on vertex
indices — no adjacency list is stored, and neighbor lookup is a single instruction.

The system is designed for DIM 5-10 (32 to 1024 neurons), the practical range for
real-world reservoir computing applications.

## Three-Stage Pipeline

```
Input -> [Reservoir] -> [TranslationLayer] -> [Readout] -> Output
          N states        2.5N features        1 prediction
```

### Stage 1: Reservoir

The reservoir (`Reservoir<DIM>`) is a DIM-dimensional Boolean hypercube with N = 2^DIM
vertices. Each vertex carries a scalar state, gathers weighted outputs from 2*DIM
neighbors via XOR-addressed connections, and applies tanh activation with a synchronous
update rule. The result: an N-dimensional time-varying state that encodes the input
history through nonlinear dynamics.

Two connection types create a rich mixing topology:
- **Hamming shells** — cumulative-bit masks reaching progressively distant vertices
- **Nearest neighbors** — single-bit flips connecting adjacent vertices

All weights are random and fixed. The spectral radius controls dynamical regime;
per-DIM optimized defaults are built in.

See [docs/Reservoir.md](docs/Reservoir.md) for full architectural details.

### Stage 2: Translation Layer

The reservoir's tanh activation encodes input history nonlinearly — powerful for
computation, but a linear readout cannot fully decode it. The translation layer
bridges this gap by expanding N raw states into 2.5N features:

| Features | Count | Purpose |
|----------|-------|---------|
| x        | N     | Raw states (linear access to reservoir) |
| x²       | N     | Magnitude information (breaks tanh symmetry) |
| x*x'     | N/2   | Cross-vertex products (quadratic interactions) |

This is a fixed algebraic transform with no learned parameters. It gives the linear
readout access to quadratic state interactions that would otherwise be invisible,
improving NRMSE by 28-83% on nonlinear tasks at DIM 7-10.

See [docs/TranslationLayer.md](docs/TranslationLayer.md) for design rationale and feature details.

### Stage 3: Readout

The readout is the only trained component. It fits a linear mapping from the 2.5N
translated features to the target signal. Two implementations are provided:

- **LinearReadout** — SGD with L2 decay and pocket selection. O(N) per sample. Preferred
  for DIM < 8.
- **RidgeRegression** — Closed-form optimal via normal equations. O(N³) solve. Preferred
  for DIM >= 8.

Both standardize features internally to handle the mixed-scale translation output.

LinearReadout also supports **streaming mode** via `TrainIncremental()` — incrementally
blending new data into the existing model without discarding learned weights. This
enables real-time adaptation for applications like process monitoring where data arrives
continuously and conditions may drift over time.

See [docs/Readout.md](docs/Readout.md) for algorithm details, selection policy, and streaming mode.

## Headline Results

All results: 3-seed average {42, 1042, 2042}, Ridge readout, full translation layer
(2.5N features), per-DIM optimized spectral radius and input scaling. MC uses Linear
readout with raw features (the standard metric).

### Mackey-Glass h=1 (chaotic time series, NRMSE, lower is better)

| DIM | N    | Raw    | Translation | vs standard ESN (0.01-0.05) |
|-----|------|--------|-------------|------------------------------|
| 5   | 32   | 0.0180 | 0.0132      | Within range                 |
| 6   | 64   | 0.0106 | 0.0082      | Beats range                  |
| 7   | 128  | 0.0061 | 0.0044      | 2-11x better                 |
| 8   | 256  | 0.0061 | 0.0040      | 3-13x better                 |
| 9   | 512  | 0.0037 | 0.0022      | 5-23x better                 |
| 10  | 1024 | 0.0028 | 0.0015      | 7-33x better                 |

### NARMA-10 (nonlinear memory, NRMSE, lower is better)

| DIM | N    | Raw   | Translation | vs standard ESN (0.2-0.4) |
|-----|------|-------|-------------|---------------------------|
| 5   | 32   | 0.566 | 0.539       | Above range                |
| 6   | 64   | 0.417 | 0.264       | Beats range                |
| 7   | 128  | 0.387 | 0.176       | Beats range                |
| 8   | 256  | 0.399 | 0.125       | Decisively beats range     |
| 9   | 512  | 0.377 | 0.072       | 3-6x better                |
| 10  | 1024 | 0.373 | 0.065       | 3-6x better                |

### Memory Capacity (sum of R², lags 1-50, Linear readout, raw features)

| DIM | N    | MC   |
|-----|------|------|
| 5   | 32   | 13.0 |
| 6   | 64   | 16.7 |
| 7   | 128  | 24.7 |
| 8   | 256  | 26.5 |
| 9   | 512  | 33.6 |
| 10  | 1024 | 33.0 |

Full results across DIM 5-10 with per-lag profiles in [diagnostics/](diagnostics/).

*Baseline ESN ranges are from Jaeger (2001) "The 'echo state' approach to analysing and
training recurrent neural networks" and Rodan & Tino (2011) "Minimum complexity echo
state networks." These represent typical results for standard random sparse ESNs with
comparable neuron counts and linear readout on raw features. HypercubeRC's advantage
comes primarily from the translation layer and feature standardization, which are
topology-independent enhancements (see [docs/DoesTopologyMatter.md](docs/DoesTopologyMatter.md)).*

## Why a Hypercube?

The hypercube topology was compared head-to-head against a random sparse ESN with
identical parameters (same N, same connection count, same weights). The result:
**equivalent performance, equivalent speed.** Topology does not determine reservoir
quality — what matters is spectral radius, connection density, and nonlinearity.

The hypercube's value is architectural:
- **Zero adjacency storage** — neighbors computed by XOR, not stored
- **O(N * DIM) per step** — sparse, not O(N²) like dense ESNs
- **Deterministic structure** — same DIM always produces the same graph
- **Hardware-friendly** — XOR addressing maps directly to gates in FPGA/ASIC

See [docs/DoesTopologyMatter.md](docs/DoesTopologyMatter.md) for the full experimental comparison.

## Building and Running

**Requirements:** C++23 compiler (GCC 13+, Clang 16+, MSVC 19.36+), CMake 3.20+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/HypercubeRC
```

The build produces four executables:

- **`HypercubeRC`** — Full benchmark suite (MC, Mackey-Glass, NARMA-10, DIM 5-10)
- **`BasicPrediction`** — Minimal example: sine wave prediction
- **`SignalClassification`** — Multi-class waveform recognition with confusion matrix
- **`StreamingAnomaly`** — Streaming anomaly detection with recovery dynamics

Start with `BasicPrediction` to see the pipeline end-to-end. Each example has a
companion `.md` file with a detailed walkthrough.

OpenMP is used for parallelism where beneficial. The build system detects MinGW,
GCC/Clang, and MSVC automatically.

## Project Structure

```
HypercubeRC/
  Reservoir.h/cpp        Hypercube reservoir (N = 2^DIM vertices)
  ESN.h                  Pipeline wrapper: Warmup, Run, collect states
  TranslationLayer.h     Feature expansion: x + x² + x*x' (2.5N features)
  SignalGenerators.h     Benchmark signal generators and NRMSE utility
  main.cpp               Benchmark suite (DIM 5-10)

  readout/
    LinearReadout.h/cpp   SGD readout with L2 decay and streaming mode
    RidgeRegression.h/cpp Closed-form optimal readout

  examples/
    BasicPrediction.cpp/md      Minimal sine wave prediction demo
    SignalClassification.cpp/md Multi-class waveform recognition
    StreamingAnomaly.cpp/md     Streaming anomaly detection

  diagnostics/
    MackeyGlass.h/md      Chaotic time series prediction
    NARMA10.h/md          Nonlinear memory benchmark
    MemoryCapacityProfile.h/md  Fading memory profile (R² vs lag)
    StateRank.h/md        Reservoir dimensionality analysis

  docs/
    Reservoir.md          Reservoir architecture and parameters
    TranslationLayer.md   Translation layer design
    Readout.md            Readout algorithms and selection policy
    DoesTopologyMatter.md Hypercube vs random sparse ESN experiment
```

## Documentation

| Document | Covers |
|----------|--------|
| [docs/Reservoir.md](docs/Reservoir.md) | Hypercube graph, connectivity, vertex model, step mechanics, parameters |
| [docs/TranslationLayer.md](docs/TranslationLayer.md) | Feature expansion rationale, antipodal pairing, standardization |
| [docs/Readout.md](docs/Readout.md) | LinearReadout and RidgeRegression algorithms, streaming mode, selection policy |
| [docs/DoesTopologyMatter.md](docs/DoesTopologyMatter.md) | Hypercube vs random ESN: does the topology actually matter? |

Diagnostic results with educational introductions are in `diagnostics/*.md`.
