# Reservoir Cascade — Extending Memory Depth

## Motivation

A single hypercube reservoir has a fixed memory horizon governed by
exponential decay at the spectral radius. For tasks requiring longer
memory — character-level language modeling, long-range time series —
this is the bottleneck (see [ReservoirMemoryBottleneck.md](ReservoirMemoryBottleneck.md)).

The cascade addresses this by stacking multiple reservoirs in series.
Each layer applies one more nonlinear transformation to the signal,
extending the effective memory depth without changing the reservoir
architecture or requiring new hyperparameters.

## Architecture

```
External input
      │
      ▼
┌──────────┐
│ Layer 0  │ ◄── InjectInput (external scalar)
│ Reservoir│
└────┬─────┘
     │ Outputs (N floats)
     │ rotated by input_rotations_[1]
     ▼
┌──────────┐
│ Layer 1  │ ◄── InjectState (rotated layer 0 output)
│ Reservoir│
└────┬─────┘
     │
     ▼
    ...
     │
     ▼
┌──────────┐
│ Layer d-1│ ◄── InjectState (rotated layer d-2 output)
│ Reservoir│
└──────────┘
     │
     ▼
  Outputs(): all layers concatenated → depth * N floats → Readout
```

**Key properties:**

- All layers share the same seed and configuration. One seed to optimize.
- No readout in the cascade loop — purely reservoir-to-reservoir coupling.
- External input enters only at layer 0. Deeper layers see progressively
  transformed versions of the signal.
- `Outputs()` concatenates all layers, giving the downstream readout
  access to multiple timescales (depth * N total features).

## Symmetry Breaking via Rotation

All reservoirs have identical weights (same seed), so feeding the same
signal into the same vertices would produce identical dynamics — no
benefit from cascading. Symmetry is broken by circularly shifting the
output array before injecting it into the next layer.

Layer `i` receives the previous layer's output rotated by
`i * (N / depth)` vertices. This spreads the rotations evenly across
the hypercube. Because the weights at each vertex are fixed by the RNG
sequence, different vertices process different parts of the signal,
creating genuinely distinct dynamics per layer despite identical weight
matrices.

Why rotation instead of different seeds:
- A single surveyed seed guarantees good reservoir dynamics everywhere.
  Multiple seeds risk a bad-seed layer degrading the cascade.
- Rotation preserves the 1D seed optimization problem. No combinatorial
  search over per-layer seeds.
- The rotation is O(N) — trivially cheap.

## Readout Geometry (Open)

The cascade produces `depth * N` output features. Two geometries are
under evaluation for how the HCNN readout consumes them:

**Option A — Depth as channels.** Reshape to `depth` input channels of
`N` spatial features. The conv stack sees the same hypercube geometry
with depth-many "views" at each position. Preserves the spatial
relationship: position `v` in layer 0 and position `v` in layer 1 are
co-located across channels. Analogous to RGB channels in image CNNs.

**Option B — Flat concatenation.** Treat `depth * N` as a single
spatial vector. Simpler, but destroys the cross-layer spatial
correspondence. The CNN must discover any inter-layer structure on
its own.

Both will be benchmarked. The choice lives entirely in readout
configuration — the cascade API is the same either way.

## API

```cpp
// Create a 3-layer cascade at DIM 10
ReservoirConfig cfg;
cfg.seed = SurveyedSeed<10>();
auto cascade = ReservoirCascade<10>::Create(3, cfg);

// Per-timestep: inject input, step, read output
cascade->InjectInput(0, input_value);
cascade->Step();
const float* out = cascade->Outputs();  // 3 * 1024 = 3072 floats

cascade->TotalOutputSize();  // depth * N
cascade->Depth();            // 3
cascade->Reset();            // zero all layers
```

## Files

| File | Role |
|------|------|
| `ReservoirCascade.h` | Class template, factory, inline accessors |
| `ReservoirCascade.cpp` | Constructor, Step, InjectInput, Outputs, Reset, explicit instantiations |
| `Reservoir.h` | `InjectState` (private, friend-accessible by cascade) |
| `Reservoir.cpp` | `InjectState` implementation — rotated overwrite of `vtx_output_` |

## Status

Reservoir-level plumbing is complete. Remaining work:
- Wire cascade into ESN as an alternative to single-Reservoir mode
- Implement readout geometry options (channels vs. flat)
- Benchmark on NARMA-10 and LM_Text to validate memory extension
