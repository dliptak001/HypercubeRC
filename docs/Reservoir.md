# Reservoir — The Hypercube Echo-State Network

## What is a reservoir?

In reservoir computing, a recurrent neural network (the "reservoir") is
used as a fixed, nonlinear dynamical system. Unlike conventional RNNs,
the reservoir's internal weights are never trained — they are set once
at initialization and frozen. Only the output layer (the "readout") is
trained, typically via simple linear regression.

This works because the reservoir transforms a 1-dimensional input signal
into an N-dimensional state vector that nonlinearly encodes the input's
recent history. The readout just needs to find the right linear
combination of those N dimensions to reconstruct the target.

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
| 5   | 32          | 10                 | 320           |
| 6   | 64          | 12                 | 768           |
| 7   | 128         | 14                 | 1,792         |
| 8   | 256         | 16                 | 4,096         |
| 9   | 512         | 18                 | 9,216         |
| 10  | 1024        | 20                 | 20,480        |

DIM is constrained to [4, 10], covering 16 to 1024 neurons — the
practical range for reservoir computing.

## Connectivity: two families of connections

Each neuron receives input from 2*DIM neighbors, organized into two
families of DIM connections each:

### Shell connections (DIM per vertex)

Cumulative-bit masks that reach progressively further across the
hypercube:

```
Mask for shell i: (1 << (i+1)) - 1
  i=0: mask=1   (flip bit 0)
  i=1: mask=3   (flip bits 0-1)
  i=2: mask=7   (flip bits 0-2)
  i=3: mask=15  (flip bits 0-3)
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
instruction per neighbor lookup.

**Note:** ShellMask(0) and NearestMask(0) both equal 1, so the first
connection in each family points to the same neighbor (bit-0 flip).
All other mask values are distinct.

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
for each nearest connection i:   s += output[v XOR nearest_mask(i)] * weight[v][DIM+i]
state[v] = tanh(alpha * s)
```

The `alpha` parameter (default 1.0) controls the steepness of the tanh
activation. Higher alpha amplifies the nonlinearity; lower alpha makes
the reservoir more linear.

### Phase 2: Synchronous swap

```
memcpy(output, state, N * sizeof(float))
```

All vertices update simultaneously. The output array is read-only
during Phase 1, so there are no race conditions — this is trivially
parallelizable with OpenMP.

## Input injection

External input is injected via `InjectInput(block, value)` **before**
each `Step()`. The input value is clamped to [-1, +1] and added to each
vertex's output via a per-vertex random projection weight (W_in):

```
output[v] += W_in[v] * clamp(input, -1, +1)
```

### Multi-input mode

For K input channels, the N vertices are block-partitioned into K
contiguous groups. Each input channel drives its own block:

- Block 0 drives vertices [0, N/K)
- Block 1 drives vertices [N/K, 2*N/K)
- ...

Each block is a subcube of the hypercube. Cross-block mixing happens
through shell connections and high-bit nearest-neighbor flips that
cross block boundaries.

## Spectral radius

The spectral radius controls the reservoir's dynamical regime:

- **Too low** (< 0.8): the reservoir forgets too quickly. Short fading
  memory, poor performance on tasks requiring history.
- **Just right** (0.85-0.95): rich dynamics with stable fading memory.
  The "edge of chaos" where reservoir computing works best.
- **Too high** (> 1.0): the reservoir becomes unstable. Small input
  differences amplify exponentially, destroying useful information.

Recurrent weights are initialized from uniform[-1,1] scaled by
1/sqrt(2*DIM), then rescaled so the spectral norm (estimated via power
iteration, up to 100 iterations with convergence check) matches the
target spectral radius. The spectral norm is the standard proxy for the
spectral radius in reservoir computing — see the comment in
`Reservoir.cpp:EstimateSpectralRadius()` for details.

## Per-DIM optimized defaults

Two sets of defaults are provided, jointly optimized on MG h=1 +
NARMA-10 + MC (3-seed average). The `FeatureMode` parameter selects
which set to use. Pass -1 for SR or input_scaling to use the auto-tuned
value.

**Raw defaults** (optimized for N-dim raw readout):

| DIM | N    | SR   | Input scaling |
|-----|------|------|---------------|
| 4   | 16   | 0.95 | 0.05          |
| 5   | 32   | 0.80 | 0.10          |
| 6   | 64   | 0.90 | 0.05          |
| 7   | 128  | 0.88 | 0.03          |
| 8   | 256  | 0.88 | 0.02          |
| 9   | 512  | 0.88 | 0.02          |
| 10  | 1024 | 0.88 | 0.02          |

**Translation defaults** (optimized for 2.5N-dim readout via TranslationLayer):

| DIM | N    | SR   | Input scaling |
|-----|------|------|---------------|
| 4   | 16   | 0.88 | 0.02          |
| 5   | 32   | 0.80 | 0.04          |
| 6   | 64   | 0.92 | 0.02          |
| 7   | 128  | 0.92 | 0.04          |
| 8   | 256  | 0.95 | 0.02          |
| 9   | 512  | 0.95 | 0.02          |
| 10  | 1024 | 0.95 | 0.02          |

**Why are translation defaults different?** Translation features (x²,
x*x') amplify the reservoir's dynamics, so the reservoir can run at
higher spectral radius (closer to instability) without losing useful
information. Input scaling is uniformly low (0.02-0.04) because the
translation layer provides the needed signal gain.

DIM 9-10 values are extrapolated from DIM 8 and not sweep-verified.
Use `Tools/StandaloneESNSweep.cpp` to run your own parameter sweeps.

## Computational properties

- **O(N * DIM) per step** — each vertex sums 2*DIM weighted neighbor
  outputs, vs. O(N²) for a dense ESN
- **Zero adjacency storage** — neighbors computed by XOR
- **Trivially parallelizable** — OpenMP over vertices with no write
  contention (all reads from the output array, all writes to the state
  array)
- **Cache-friendly** — XOR addressing produces structured, predictable
  access patterns

See `docs/DoesTopologyMatter.md` for measured timing vs. a random sparse ESN
(equivalent results, equivalent speed at DIM 5-10, but the hypercube
needs no adjacency storage).
