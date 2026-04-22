# HypercubeRC

[![Build wheels](https://github.com/dliptak001/HypercubeRC/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeRC/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-rc)](https://pypi.org/project/hypercube-rc/)

A reservoir computer whose neurons live on a Boolean hypercube — a
DIM-dimensional graph where each vertex is addressed by a DIM-bit binary
index, with all connectivity defined by XOR operations on vertex indices.
No adjacency list is stored. N = 2^DIM neurons, DIM 5-16.

Licensed under the [Apache License 2.0](LICENSE).

## What is Reservoir Computing?

Reservoir computing is a machine learning paradigm for temporal data. Instead of
training a recurrent neural network end-to-end (expensive, unstable, prone to
vanishing gradients), reservoir computing splits the problem in two:

1. **A fixed, random recurrent network (the reservoir)** receives input and
   produces a high-dimensional, time-dependent state. Its weights are never
   trained — they are set once at initialization and frozen.

2. **A simple linear readout** learns to map the reservoir's state to the desired
   output. This is a single-layer linear regression — fast, stable, and globally
   optimal.

The insight is that a sufficiently rich dynamical system, driven by input, will
naturally create a high-dimensional embedding of the input history. The nonlinear
recurrence provides the computational power; the linear readout provides the
learning. This separation makes reservoir computing orders of magnitude faster to
train than backpropagation-based RNNs, while achieving competitive performance on
tasks that require memory and nonlinear computation.

## What is HypercubeRC?

HypercubeRC is a reservoir computing architecture where the reservoir's
connectivity is defined by a Boolean hypercube of dimension DIM, giving N = 2^DIM
neurons. Each neuron receives from 2*DIM - 2 neighbors via two connection
families — cumulative-bit Hamming shells for long-range mixing and single-bit
nearest-neighbor flips for local coupling. All neighbor addresses are computed by
XOR on vertex indices; no adjacency list is stored.

The system targets DIM 5-16 (32 to 65536 neurons), the practical range for
reservoir computing applications.

## Scale-Invariant Hyperparameters

The general-purpose spectral radius (0.90) and input scaling (0.02) are
**independent of reservoir size**. The same two values produce the best
general-purpose results at every DIM from 5 to 9 (32 to 512 neurons), verified
across NARMA-10 benchmarks with per-DIM
optimal seeds (selected by 500-seed survey) and three-pass grid sweeps (coarse,
normal, fine).

This is unusual in reservoir computing. Standard practice requires re-sweeping
the spectral radius whenever the reservoir size changes, because the stability
threshold of the weight matrix shifts with N in random sparse networks.
Practitioners routinely report per-size tuning, and published ESN results almost
always include a per-N hyperparameter table. The hypercube breaks this pattern.

**Why it works.** The hypercube is *vertex-transitive*: every vertex has exactly
the same local neighborhood structure — same degree, same shell distances, same
symmetry group. Increasing DIM adds more vertices, but each one sees the same
number of neighbors with the same weight distribution. The per-vertex dynamics
are structurally identical at every scale, so the stability threshold — the
spectral radius where dynamics transition from convergent to chaotic — is a
property of the local structure, not the global size.

Random sparse networks lack this property. Their heterogeneous degree
distributions mean some neurons are hubs, others peripheral. This heterogeneity
interacts with spectral radius differently at each N, forcing per-size
retuning.

**Practical consequence: configure once, scale freely.** Set SR=0.90 and
input_scaling=0.02, then increase DIM to add capacity without re-sweeping. No
per-size tuning tables, no factory functions, no search. The most common
hyperparameter burden in reservoir computing is eliminated.

**Of academic interest.** Scale invariance of optimal hyperparameters is a direct
consequence of vertex-transitivity — a graph-theoretic property absent from the
random sparse topologies used in conventional reservoir computing. To our
knowledge, this is the first observation that a vertex-transitive reservoir
topology produces size-independent optimal spectral radius and input scaling. The
finding suggests that reservoir topology studies should examine the relationship
between graph symmetry and hyperparameter sensitivity more broadly. Vertex-
transitive graphs (hypercubes, tori, Cayley graphs) may constitute a class of
reservoir substrates with fundamentally different tuning characteristics than
random graphs.

See [docs/ScaleInvariance.md](docs/ScaleInvariance.md) for sweep data and
analysis.

## Seed Quality is Topology-Invariant

A 500-seed survey across NARMA-10 and Memory Capacity
benchmarks (DIM 5-8, with library support through DIM 16) shows that the rank ordering of seeds by performance
is stable across hyperparameter configurations. Seeds screened at one
(SR, input_scaling) pair rank similarly at any other pair within the
operating range.

**Input scaling has no effect on seed ranking.** Spearman rank correlation
across IS values {0.010, 0.015, 0.020, 0.025, 0.030} at fixed SR=0.90
is rho >= 0.949 at all DIM values. This is expected — input scaling is a
linear gain that cannot reorder the relative quality of weight topologies.

**Spectral radius has a moderate, task-dependent effect.** Within the
0.85-0.90 operating corridor, Spearman rho exceeds 0.82 for all three
benchmarks at all DIM values. Correlation decays with SR distance and
collapses at SR=1.00 (edge of chaos), where NARMA-10 even goes negative
at DIM 7 — low-SR and high-SR regimes select for opposite topological
properties.

**SR=0.90 is the best general-purpose default.** Memory Capacity favors
0.90-0.95, but the improvement from 0.90→0.95 comes with doubled
variance. NARMA-10 hits minimum at 0.90-0.95. SR=0.90 gives near-optimal
mean performance on both tasks with the lowest variance and strongest
rank correlation.

The practical consequence: screen seeds once at SR=0.90, IS=0.02 and
reuse the winners at any configuration. Seed quality is an intrinsic
property of the weight topology, not an artifact of a specific
hyperparameter setting.

## Why a Hypercube?

A controlled head-to-head experiment
([docs/DoesTopologyMatter.md](docs/DoesTopologyMatter.md)) showed that a random
sparse ESN with identical parameters produces equivalent benchmark performance
and equivalent speed at DIM 5-12. The hypercube does not compute better than a
random graph — but it computes more elegantly, and its vertex-transitive symmetry
produces the scale-invariant hyperparameters described above.

**Zero storage overhead.** No adjacency list. The reservoir needs only two arrays
(states and weights); a random graph adds a third (adjacency). At DIM=10 this is
80 KB — trivial on a modern CPU, but it scales as O(N * DIM) additional memory
and becomes relevant at scale or in memory-constrained environments.

**Implicit, reproducible structure.** XOR addressing is deterministic. Two
implementations with the same DIM automatically agree on connectivity — no
serialization, no graph exchange, no compatibility concerns. The entire topology
is implicit in the binary representation of vertex indices.

**Trivial parallelization.** XOR-based neighbor computation has no data
dependencies beyond the read-only output array. The update loop is trivially
SIMD-able with zero synchronization overhead.

**Direct hardware mapping.** In FPGA or neuromorphic implementations, XOR
addressing maps to gates directly — no routing table, no memory controller for
adjacency lookup. The hypercube is one of the few reservoir topologies that can
be implemented without a stored graph in dedicated hardware.

**Scale-invariant tuning.** Unlike random sparse graphs, the hypercube's
vertex-transitive topology produces size-independent optimal hyperparameters
(see above). This is a property of the graph symmetry, not available to random
topologies.

**Topology-invariant seed quality.** The deterministic structure means that seed
quality — how well a particular random weight draw performs — is a stable property
of the topology, not an artifact of a specific hyperparameter setting. Seeds
screened once at the default SR and input scaling rank the same way across the
operating range (Spearman rho >= 0.82 within the SR corridor, rho >= 0.949 across
input scaling). Screen once, reuse everywhere — no per-configuration seed search.

None of the storage or addressing advantages produce a measurable speed
difference in a software benchmark at DIM 5-12 on a modern CPU with deep cache
hierarchies. They become relevant at scale or in constrained environments.

## Architecture Summary

| Property | Detail                                                          |
|----------|-----------------------------------------------------------------|
| Neurons | N = 2^DIM, DIM 5-16 (32 to 65536)                               |
| Connections per neuron | 2*DIM - 2 (DIM-2 shells + DIM nearest-neighbor)                 |
| Addressing | XOR on vertex indices — O(1) per lookup, zero storage           |
| Step cost | O(N * DIM) per timestep — sparse, not O(N²)                     |
| Hyperparameters | Scale-invariant: SR=0.90, input_scaling=0.02, all DIM           |
| Readout cost control | `output_fraction` selects M of N vertices for readout           |
| Multi-input | Stride-interleaved: channel k drives vertices k, k+K, k+2K, ... |

**Output selection.** The `output_fraction` parameter controls how many vertices
feed the readout. At the default (1.0), all N vertices contribute. At 0.5, a
stride-selected subset of N/2 vertices is used. This decouples reservoir size
from readout cost.

## Pipeline

```
Input ──> Reservoir ──> HCNNReadout ──> Output
           N states      trained
```

### Stage 1: Reservoir

The reservoir (`Reservoir<DIM>`) is a DIM-dimensional Boolean hypercube with
N = 2^DIM vertices. Each vertex gathers weighted outputs from 2*DIM - 2
neighbors via XOR-addressed connections and applies a leaky integrator with tanh
activation:

```
activation = tanh(alpha * weighted_sum)
state[v] = (1 - leak_rate) * old_state[v] + leak_rate * activation
```

Two connection families create rich mixing:
- **Hamming shells** (DIM-2 per vertex) — cumulative-bit masks reaching
  progressively distant vertices, skipping distance-1 and antipodal
- **Nearest neighbors** (DIM per vertex) — single-bit flips connecting adjacent
  vertices along every dimension

All weights are random and fixed. See [docs/Reservoir.md](docs/Reservoir.md) for
full architectural details.

### Stage 2: Readout

The readout is the only trained component — a mapping from raw reservoir
states to the target signal:

- **HCNNReadout** — HypercubeCNN learned convolutional readout on raw
  reservoir state. Discovers nonlinear feature interactions via learned
  convolution kernels on the hypercube topology. Supports regression
  (single/multi-output), multi-class classification, and online training.

See [docs/HCNNReadout.md](docs/HCNNReadout.md) for algorithm details,
architecture auto-sizing, and streaming mode.

## Headline Results

### NARMA-10 (nonlinear memory, NRMSE, lower is better)

All results: per-DIM surveyed seed, HCNN baseline config
(`HRCCNNBaseline<DIM>()`), scale-invariant defaults (SR=0.90, input_scaling=0.02).

| DIM | N    | HCNN NRMSE |
|-----|------|------------|
| 7   | 128  | 0.218      |
| 8   | 256  | 0.153      |
| 9   | 512  | 0.134      |
| 10  | 1024 | 0.122      |

Standard ESN baseline: 0.2-0.4 (Jaeger 2001, Rodan & Tino 2011).

## Related Work

Katori (2019) "[Reservoir Computing Based on Dynamics of Pseudo-Billiard System
in Hypercube](https://ieeexplore.ieee.org/document/8852329/)" (IJCNN 2019, Best
Paper Award) also applies hypercube structure to reservoir computing. Katori's
approach uses pseudo-billiard chaotic dynamics within a hypercube state space,
where units interact through binary states via a Chaotic Boltzmann Machine.
HypercubeRC takes a different approach: the hypercube defines the connectivity
graph of an echo-state network, with XOR-addressed wiring between tanh neurons.
The two architectures share the hypercube as a structural primitive but differ in
dynamics, activation model, and the role the hypercube plays.

## Python SDK

Pre-built wheels are available on [PyPI](https://pypi.org/project/hypercube-rc/)
for Python 3.10-3.13 on Windows (x64), Linux (x86_64, aarch64), and macOS
(x86_64, arm64). No compiler required.

```bash
pip install hypercube-rc
```

```python
import numpy as np
import hypercube_rc as hrc

signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
esn = hrc.ESN(dim=7, seed=42)
esn.fit(signal, warmup=200)
print(f"R2 = {esn.r2():.6f}")
```

See [docs/Python_SDK.md](docs/Python_SDK.md) for the full API reference.

## Building and Running (C++)

**Requirements:** C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+), CMake 4.1+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/HypercubeRC
```

The build produces six executables:

| Target | Purpose |
|--------|---------|
| `HypercubeRC` | Benchmark suite: NARMA-10 across DIM (library supports DIM 5-16) |
| `BasicPrediction` | Minimal example: sine wave prediction |
| `SignalClassification` | Multi-class waveform recognition with confusion matrix |
| `StreamingAnomaly` | Streaming anomaly detection with recovery dynamics |
Start with `BasicPrediction` to see the pipeline end-to-end. Each example has a
companion `.md` file with a detailed walkthrough.

## Project Structure

```
HypercubeRC/
  Reservoir.h/cpp        Hypercube reservoir (N = 2^DIM vertices)
  ESN.h/cpp              Unified pipeline: warmup, run, train, predict
  main.cpp               Benchmark suite entry point (DIM 5-8)
  docs/CPP_SDK.md        C++ consumer documentation for the static library

  readout/
    HCNNReadout.h/cpp     Learned convolutional readout (PIMPL)
    HCNNPresets.h         Per-DIM baseline configs and surveyed seeds

  examples/
    BasicPrediction.cpp/md      Minimal sine wave prediction
    SignalClassification.cpp/md Multi-class waveform recognition
    StreamingAnomaly.cpp/md     Streaming anomaly detection

  diagnostics/
    BenchmarkSuite.h      Orchestrates NARMA-10 across DIM
    NARMA10.h/md          Nonlinear memory benchmark
    SignalGenerators.h    Benchmark signal generators and NRMSE utility
    StateRank.h/md        Reservoir dimensionality and input correlation

  docs/
    Reservoir.md          Reservoir architecture, connectivity, parameters
    HCNNReadout.md        HCNN readout: architecture, training, streaming mode
    ScaleInvariance.md    Scale-invariant hyperparameters: evidence and analysis
    DoesTopologyMatter.md Hypercube vs random sparse ESN experiment
    Tuning.md             Practical tuning guide: parameters, scenarios, workflow
```

## Documentation

| Document | Covers |
|----------|--------|
| [docs/Reservoir.md](docs/Reservoir.md) | Hypercube graph, connectivity, leaky integrator, spectral radius, scale-invariant defaults |
| [docs/HCNNReadout.md](docs/HCNNReadout.md) | HCNN readout architecture, training algorithm, streaming mode, ESN interface |
| [docs/ScaleInvariance.md](docs/ScaleInvariance.md) | Why SR=0.90 and input_scaling=0.02 work at every DIM — sweep data and vertex-transitivity analysis |
| [docs/DoesTopologyMatter.md](docs/DoesTopologyMatter.md) | Hypercube vs random ESN: equivalent performance, different architectural tradeoffs |
| [docs/Tuning.md](docs/Tuning.md) | Practical tuning guide: parameter-by-parameter advice, common scenarios, workflow |
| [docs/Python_SDK.md](docs/Python_SDK.md) | Python SDK: pip install, fit/predict API, streaming, persistence |
| [docs/CPP_SDK.md](docs/CPP_SDK.md) | C++ static library: build, install, find_package usage, API reference |

Diagnostic `.md` files in `diagnostics/` provide educational introductions to
each benchmark with sample results and interpretation guidance.
