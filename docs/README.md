# Documentation Guide

This directory contains detailed documentation for each component of
HypercubeRC. If you're new to the project, start with the
[project README](../README.md) for an overview, then follow the reading
order below.

## Suggested reading order

### 1. Understand the architecture

| Document | What you'll learn |
|----------|-------------------|
| [Reservoir.md](Reservoir.md) | How the hypercube reservoir works — topology, connectivity, timestep mechanics, spectral radius, scale-invariant defaults |
| [TranslationLayer.md](TranslationLayer.md) | Why raw tanh states limit a linear readout, and how the translation layer breaks through that bottleneck |
| [Readout.md](Readout.md) | The two readout algorithms (SGD vs. Ridge), when to use each, feature standardization, and streaming mode |

These three documents cover the full pipeline:

```
Input ──> Reservoir (N states) ──> Translation (2.5N features) ──> Readout ──> Prediction
           [Reservoir.md]          [TranslationLayer.md]          [Readout.md]
```

### 2. See it in action

The `examples/` directory contains three worked examples, each with a
companion `.md` walkthrough:

| Example | What it demonstrates |
|---------|---------------------|
| [BasicPrediction](../examples/BasicPrediction.md) | Simplest end-to-end demo — predict a sine wave. Start here. |
| [SignalClassification](../examples/SignalClassification.md) | Reservoir as feature extractor for multi-class waveform recognition |
| [StreamingAnomaly](../examples/StreamingAnomaly.md) | Anomaly detection in a simulated industrial process |

### 3. Understand the benchmarks

The `diagnostics/` directory contains four standard RC benchmarks, each
with educational `.md` documentation:

| Diagnostic | What it measures |
|------------|-----------------|
| [MackeyGlass](../diagnostics/MackeyGlass.md) | Chaotic time series prediction (tests nonlinear dynamics tracking) |
| [NARMA10](../diagnostics/NARMA10.md) | Combined memory + nonlinear computation (the hardest standard benchmark) |
| [MemoryCapacityProfile](../diagnostics/MemoryCapacityProfile.md) | How far back the reservoir can remember (fading memory profile) |
| [StateRank](../diagnostics/StateRank.md) | Reservoir dimensionality and input correlation analysis |

### 4. Go deeper

| Document | What it covers |
|----------|---------------|
| [ScaleInvariance.md](ScaleInvariance.md) | Why SR=0.90 and input_scaling=0.02 are optimal at every DIM — the hypercube's vertex-transitive topology |
| [DoesTopologyMatter.md](DoesTopologyMatter.md) | Head-to-head experiment: hypercube vs. random sparse ESN — same performance, different tradeoffs |

## Key source files

For readers who prefer to learn from code, the class-level doc comments
in the header files are written for an educational audience:

| Header | Class/Functions |
|--------|----------------|
| `ESN.h` | `ESN<DIM>` — the pipeline wrapper (warmup, run, collect states) |
| `Reservoir.h` | `Reservoir<DIM>` — the hypercube reservoir core |
| `TranslationLayer.h` | `TranslationTransform<DIM>()`, `TranslationTransformSelected<DIM>()` — feature expansion |
| `diagnostics/SignalGenerators.h` | `GenerateMackeyGlass()`, `GenerateNARMA10()`, `ComputeNRMSE()` |
| `readout/LinearReadout.h` | `LinearReadout` — SGD readout with streaming support |
| `readout/RidgeRegression.h` | `RidgeRegression` — closed-form optimal readout |
