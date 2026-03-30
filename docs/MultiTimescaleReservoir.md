# Multi-Timescale Reservoir

## Motivation

A standard reservoir has one uniform dynamical regime: all neurons share the
same spectral radius and update rule, so they all "forget" at roughly the same
rate. This is a trade-off — a high spectral radius gives long memory but risks
instability; a low one is stable but forgetful.

Multi-timescale reservoirs remove this trade-off by creating **regions with
different time constants within the same network**. Some neurons are fast and
reactive (short memory, high sensitivity to recent input), others are slow and
sticky (long memory, smoothed response). The readout layer then gets to pick
from both fast recent context and slow long-term context simultaneously.

This is the central idea behind the `multi-timescale-reservoir` branch.

## Background: Leaky Integrator Neurons

The current reservoir (`Reservoir.cpp:87`) uses a full-replacement update:

```
state[v] = tanh(alpha * s)
```

A **leaky integrator** neuron blends its old state with the new activation:

```
state[v] = (1 - leak) * old_state[v] + leak * tanh(alpha * s)
```

The **leak rate** (in (0, 1]) controls the neuron's time constant:

| Leak Rate | Behavior |
|-----------|----------|
| 1.0       | Full replacement (current behavior, no memory) |
| 0.5       | Half old state, half new — moderate smoothing |
| 0.1       | 90% old state retained — very slow, long memory |

A leak rate of 1.0 recovers the current update rule exactly, so this is a
strict generalization with no performance cost at the default setting.

## Design

### Two Timescale Zones via the Hypercube

The Boolean hypercube has a natural partition: the **highest bit** of each
vertex index. This splits N = 2^DIM vertices into two equal halves:

- **Fast zone** (bit DIM-1 = 0): vertices `[0, N/2)` — high leak rate
- **Slow zone** (bit DIM-1 = 1): vertices `[N/2, N)` — low leak rate

This partition is topologically meaningful. Shell and nearest-neighbor
connections cross the zone boundary (the highest-bit flip is one of the
nearest-neighbor connections), so information flows between zones without
any special coupling mechanism. The hypercube's existing connectivity
provides inter-zone mixing for free.

### Parameters

Two new parameters on `Reservoir::Create()`:

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `leak_fast` | float | 1.0 | Leak rate for the fast zone (vertices 0..N/2-1) |
| `leak_slow` | float | 1.0 | Leak rate for the slow zone (vertices N/2..N-1) |

Both default to 1.0, which recovers the current full-replacement behavior
exactly. No existing code changes behavior unless these are explicitly set.

### Modified Update Rule

The current `UpdateState()`:

```cpp
void Reservoir<DIM>::UpdateState(size_t v)
{
    const float* w = vtx_weight_.data() + v * NUM_CONNECTIONS;
    float s = 0.0f;
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[v ^ ShellMask(i)] * w[i];
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[v ^ NearestMask(i)] * w[DIM + i];
    vtx_state_[v] = std::tanh(alpha_ * s);
}
```

Becomes:

```cpp
void Reservoir<DIM>::UpdateState(size_t v)
{
    const float* w = vtx_weight_.data() + v * NUM_CONNECTIONS;
    float s = 0.0f;
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[v ^ ShellMask(i)] * w[i];
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[v ^ NearestMask(i)] * w[DIM + i];

    const float activation = std::tanh(alpha_ * s);
    const float leak = (v < N / 2) ? leak_fast_ : leak_slow_;
    vtx_state_[v] = (1.0f - leak) * vtx_output_[v] + leak * activation;
}
```

The branch condition `v < N/2` compiles to a single comparison — no lookup
table, no per-vertex storage. The two-zone split costs one compare and one
multiply-add per neuron per step.

### Relationship Between Spectral Radius and Leak Rate

SR and leak rate control different things and cannot substitute for each other:

- **Spectral radius** controls the **gain** of the recurrent signal — how
  large `tanh(alpha * s)` is at each timestep.
- **Leak rate** controls **how much of that signal is applied** — how fast
  the neuron moves toward the new activation.

A low SR makes the new activation *small*. A low leak rate makes the neuron
*ignore most of the new activation regardless of its size*. You can't get a
slow, sticky neuron by lowering SR — that just makes the activation weaker,
but the neuron still fully replaces its state every step. Conversely, you
can't get a strong recurrent signal by raising the leak rate — that just
makes the neuron respond faster to whatever signal SR already produced.

They combine into an **effective spectral radius** for the linearized system:

```
rho_eff = (1 - a) + a * rho(W)
```

At leak=1.0 (current behavior), `rho_eff = rho(W)` — SR is all that matters.
At leak=0.3 with SR=0.9, `rho_eff = 0.7 + 0.27 = 0.97` — the slow leak
raises the effective SR toward 1.0, giving longer memory.

In practice, **lower leak rates tolerate higher spectral radii**. The slow
zone can safely run at SR > 1.0 because the leak dampens oscillations.

The current per-DIM SR defaults were tuned assuming leak=1.0. Once a slow
zone is added with leak < 1.0, those neurons will have a higher effective SR
than intended. The sweeps should co-optimize SR and leak_slow jointly.

### Implementation Order

The leaky integrator is a **prerequisite** for multi-timescale reservoirs.
The multi-timescale idea is just "assign different leak rates to different
zones" — but that requires the leak mechanism to exist first.

1. **Add leaky integrator to `UpdateState()`** — single global leak rate,
   defaulting to 1.0 (no behavior change).
2. **Split into per-zone leak rates** — fast zone / slow zone with different
   values (the multi-timescale feature itself).

### Per-DIM Default Leak Rates

Initial values (to be refined by sweep):

| DIM | leak_fast | leak_slow | Rationale |
|-----|-----------|-----------|-----------|
| 4-5 | 1.0       | 0.5       | Small reservoirs — moderate separation |
| 6-7 | 1.0       | 0.3       | Enough neurons for meaningful zones |
| 8-10| 1.0       | 0.2       | Large reservoirs — aggressive slow zone |

The fast zone stays at 1.0 (full replacement) to preserve the short-term
sensitivity that currently works well. Only the slow zone changes.

## Impact on Downstream Components

### Translation Layer

No changes needed. The translation layer operates on the N-dimensional output
vector regardless of how those outputs were computed. The slow-zone neurons
will naturally produce smoother, more correlated features — the x^2 and x*x'
terms in the translation layer will capture cross-zone interactions.

### Readout

No changes needed. The readout sees the same N (or 2.5N) features. The
ridge regression or SGD will learn to weight fast-zone and slow-zone neurons
differently, which is exactly the goal — the readout *selects* the timescale
mix that best fits the task.

### ESN Wrapper

No changes needed. Warmup, Run, and state collection are agnostic to the
update rule internals.

### Diagnostics

The three existing benchmarks are well-suited to validate this feature:

- **Mackey-Glass**: Chaotic prediction benefits from long memory (slow zone)
  combined with fast tracking of recent dynamics (fast zone).
- **NARMA-10**: Nonlinear autoregressive task with 10-step dependencies.
  The slow zone should improve the model's ability to retain the needed
  history without increasing N.
- **Memory Capacity**: Directly measures how far back the reservoir can
  recall past inputs. This is the primary metric — multi-timescale should
  increase total memory capacity by adding long-delay recall from the slow
  zone without sacrificing short-delay recall from the fast zone.

## Implementation Plan

1. **Add `leak_fast_` and `leak_slow_` members** to `Reservoir<DIM>`.
   Default both to 1.0 in the constructor. Add parameters to `Create()`.

2. **Modify `UpdateState()`** with the leaky integrator formula and
   zone-conditional leak rate (as shown above).

3. **Add per-DIM defaults** for leak rates, following the same pattern as
   `RawSpectralRadius()` / `TranslationSpectralRadius()`.

4. **Run the benchmark suite** at each DIM to validate:
   - leak_fast=1.0, leak_slow=1.0 reproduces current results exactly.
   - leak_fast=1.0, leak_slow=0.3 improves Memory Capacity without
     destroying MG/NARMA-10 performance.

5. **Sweep leak_slow** over {0.1, 0.2, 0.3, 0.5, 0.7} at each DIM to find
   optimal defaults.

6. **Document results** in the benchmark tables.

## Future Extensions

- **Per-zone spectral radius**: Scale the slow-zone weights to a different
  target SR than the fast zone, allowing independent tuning of stability
  and memory in each region.

- **More than two zones**: Use multiple bits to create 4 or 8 timescale
  bands. The hypercube's bit-addressable structure makes K-zone partitions
  natural (use the top log2(K) bits).

- **Continuous leak rates**: Instead of two discrete zones, assign each
  vertex a leak rate as a smooth function of its index (e.g., linearly
  interpolated from fast to slow across the vertex space).

- **Learnable leak rates**: Treat leak rates as trainable parameters
  optimized alongside the readout weights.
