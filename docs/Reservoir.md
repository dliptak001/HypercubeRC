# Reservoir — The Hypercube Echo-State Network

## What is a reservoir?

In reservoir computing, a recurrent neural network (the "reservoir") is
used as a fixed, nonlinear dynamical system. Unlike conventional RNNs,
the reservoir's internal weights are never trained — they are set once
at initialization and frozen. Only the output layer (the "readout") is
trained.

This works because the reservoir transforms a 1-dimensional input signal
into an N-dimensional state vector that nonlinearly encodes the input's
recent history. The readout learns to extract the right features from
those N dimensions to reconstruct the target. Traditional reservoir
computing uses a linear readout (ridge regression); HypercubeRC uses a
learned convolutional readout (HCNN) that operates directly on the
hypercube state and discovers nonlinear features automatically — see
[HCNNReadout.md](HCNNReadout.md).

The quality of the reservoir depends on two things:
1. **Rich dynamics** — different inputs produce distinguishably different
   state trajectories.
2. **Fading memory** — the state reflects recent input history, not
   ancient history or initial conditions.

## What makes this reservoir different: the hypercube topology

Most echo-state networks use random sparse connectivity. HypercubeRC
arranges its N = 2^DIM neurons on a **Boolean hypercube** — a
DIM-dimensional graph where each vertex is addressed by a DIM-bit binary
index, and two vertices are neighbors when their indices differ by
exactly one bit.

The key insight: neighbor addresses are computed by XOR operations on
vertex indices. No adjacency list is stored. The entire graph structure
is implicit in the binary representation of the vertex addresses.

| DIM | N (neurons) | Connections/neuron | Total weights |
|-----|-------------|--------------------|---------------|
| 5   | 32          | 8                  | 256           |
| 6   | 64          | 10                 | 640           |
| 7   | 128         | 12                 | 1,536         |
| 8   | 256         | 14                 | 3,584         |
| 9   | 512         | 16                 | 8,192         |
| 10  | 1024        | 18                 | 18,432        |

DIM is constrained to [5, 16], covering 32 to 65,536 neurons — the
practical range for reservoir computing.

## Connectivity: two families of connections

Each neuron receives input from 2*DIM - 2 neighbors, organized into two
families:

### Shell connections (DIM-2 per vertex)

Cumulative-bit masks that reach progressively further across the
hypercube. Distance-1 (mask=1, same as nearest-neighbor) and the
antipodal shell (mask=N-1, all bits set) are skipped:

```
ShellMask(k) = (1 << (k+1)) - 1, for k = 1 to DIM-2
  k=1: mask=3    (flip bits 0-1)
  k=2: mask=7    (flip bits 0-2)
  k=3: mask=15   (flip bits 0-3)
  ...
```

Each successive shell connection scrambles more low-order bits,
connecting to vertices at increasing Hamming distance. This creates
long-range information mixing across the hypercube.

### Nearest-neighbor connections (DIM per vertex)

Single-bit flips — the standard hypercube edges:

```
Mask for neighbor i: 1 << i
  i=0: mask=1   (flip bit 0)
  i=1: mask=2   (flip bit 1)
  i=2: mask=4   (flip bit 2)
  i=3: mask=8   (flip bit 3)
  ...
```

These connect each vertex to its DIM Hamming-distance-1 neighbors,
providing local coupling along every dimension.

Both mask types are computed inline from the loop index — a single XOR
instruction per neighbor lookup. All mask values across both families
are distinct (shells start at mask=3 precisely to avoid overlapping
with the nearest-neighbor bit-0 flip at mask=1).

### Why two connection types?

The combination provides both **local coupling** (nearest-neighbor) and
**long-range mixing** (shells). Nearest-neighbor connections propagate
information along individual dimensions; shell connections mix
information across multiple dimensions simultaneously. Together, they
ensure that input injected at any vertex can influence the entire
reservoir within a few steps.

## How a timestep works

Each vertex carries two scalars:

| Field | Purpose |
|-------|---------|
| `vtx_state_[v]` | Internal state — updated each step |
| `vtx_output_[v]` | Published output — what neighbors read |

The separation is what makes **synchronous update** work: all vertices
read from the output array (last step's values) and write to the state
array (this step's values). No vertex sees a partially-updated state.

### Phase 1: Compute new states (parallel over vertices)

For each vertex v:
```
s = 0
for each shell connection i:     s += output[v XOR shell_mask(i)] * weight[v][i]
for each nearest connection i:   s += output[v XOR nearest_mask(i)] * weight[v][DIM-2+i]
activation = tanh(alpha * s)
state[v] = (1 - leak_rate) * output[v] + leak_rate * activation
```

The `alpha` parameter (default 1.0) controls the steepness of the tanh
activation. Higher alpha amplifies the nonlinearity; lower alpha makes
the reservoir more linear. The `leak_rate` parameter (default 1.0)
controls how quickly neurons replace their state: at 1.0, the old state
is fully replaced; at lower values (e.g., 0.3), 70% of the old state
persists, creating a leaky integrator that smooths dynamics and extends
temporal memory.

### Phase 2: Publish new states

```
memcpy(output, state, N * sizeof(float))
```

All vertices update simultaneously. The output array is read-only
during Phase 1, so there are no race conditions.

## Input injection

External input is injected via `InjectInput(channel, value)` **before**
each `Step()`. The input value is clamped to [-1, +1] and added to each
vertex's output via a per-vertex random projection weight (W_in):

```
output[v] += W_in[v] * clamp(input, -1, +1)
```

### Multi-input mode

For K input channels (`num_inputs = K`), each channel drives a
stride-interleaved subset of the N vertices:

- Channel 0 drives vertices 0, K, 2K, 3K, ...
- Channel 1 drives vertices 1, K+1, 2K+1, 3K+1, ...
- Channel k drives vertices k, k+K, k+2K, ...

Each channel's vertices are uniformly distributed across the hypercube,
ensuring every channel has equal coverage of the topology. Cross-channel
mixing happens naturally through the recurrent connections (shells and
nearest-neighbor flips span across channels).

## Spectral radius

The spectral radius controls the reservoir's dynamical regime:

- **Too low** (< 0.8): the reservoir forgets too quickly. Short fading
  memory, poor performance on tasks requiring history.
- **Just right** (0.85-0.95): rich dynamics with stable fading memory.
  The "edge of chaos" where reservoir computing works best.
- **Too high** (> 1.0): the reservoir becomes unstable. Small input
  differences amplify exponentially, destroying useful information.

Recurrent weights are initialized from uniform[-1,1] scaled by
1/sqrt(2*DIM - 2), then rescaled so the spectral norm (estimated via power
iteration, up to 100 iterations with convergence check) matches the
target spectral radius. The spectral norm is the standard proxy for the
spectral radius in reservoir computing — see the comment in
`Reservoir.cpp:EstimateSpectralRadius()` for details.

## Scale-invariant defaults

The `ReservoirConfig` struct provides universal defaults that work
across all DIM values:

| Parameter | Default | Notes |
|-----------|---------|-------|
| spectral_radius | 0.90 | Edge-of-chaos optimum |
| input_scaling | 0.02 | Uniform W_in scaling |

These defaults are **scale-invariant** — the same values are optimal at
every DIM. This is a consequence of the hypercube's vertex-transitive
topology: every vertex has an identical local neighborhood structure
(same degree, same shell distances, same symmetry), so the dynamics
that produce good reservoir computing at one scale produce good dynamics
at every scale. Empirically verified at DIM 5-9 via three-pass grid
sweeps; the vertex-transitive property guarantees the same holds at
higher DIM.

No per-DIM lookup tables or factory functions are needed. Just use
`ReservoirConfig{}` and the defaults are correct. See
[ScaleInvariance.md](ScaleInvariance.md) for the full sweep data and
analysis.

## Computational properties

- **O(N * DIM) per step** — each vertex sums 2*DIM - 2 weighted neighbor
  outputs, vs. O(N²) for a dense ESN
- **Zero adjacency storage** — neighbors computed by XOR
- **Trivially parallelizable** — no write contention (all reads from
  the output array, all writes to the state array)
- **Cache-friendly** — XOR addressing produces structured, predictable
  access patterns

See `docs/DoesTopologyMatter.md` for measured timing vs. a random sparse ESN
(equivalent results, equivalent speed at DIM 5-10, but the hypercube
needs no adjacency storage).
