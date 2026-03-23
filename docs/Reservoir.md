# Reservoir

## Concept

The Reservoir class (`Reservoir<DIM>`) implements a continuous echo-state reservoir on a
DIM-dimensional Boolean hypercube graph with N = 2^DIM vertices. Each vertex carries a
scalar floating-point state, gathers weighted outputs from its neighbors through XOR-based
addressing, applies tanh activation with configurable leak rate, and publishes its new
state in a synchronous swap.

The hypercube topology means neighbor addresses are computed by XOR operations on vertex
indices — no adjacency list is needed, and neighbor lookup is a single XOR instruction.

## Hypercube Substrate

The graph is a DIM-dimensional Boolean hypercube with N = 2^DIM vertices. DIM is
constrained to [5, 10], covering 32 to 1024 vertices — the practical deployment range
for reservoir computing.

| DIM | N    | Connections/vertex | Total weights |
|-----|------|--------------------|---------------|
| 5   | 32   | 10                 | 320           |
| 6   | 64   | 12                 | 768           |
| 7   | 128  | 14                 | 1,792         |
| 8   | 256  | 16                 | 4,096         |
| 9   | 512  | 18                 | 9,216         |
| 10  | 1024 | 20                 | 20,480        |

## Connectivity

Each vertex has 2*DIM connections, split into two types:

**Hamming-shell connections (DIM):** Cumulative-bit selectors that reach progressively
more distant vertices on the hypercube.
- Mask for connection i: `(1 << (i+1)) - 1` producing masks 1, 3, 7, 15, 31, ...
- These create "shells" of increasing Hamming distance from each vertex.

**Nearest-neighbor connections (DIM):** Single-bit flips that connect to adjacent vertices
at Hamming distance 1.
- Mask for connection i: `1 << i` producing masks 1, 2, 4, 8, 16, ...
- These are the standard hypercube edges.

Both mask types are computed inline from the loop index — no stored adjacency. Each
vertex has fully independent weights (N * 2*DIM total recurrent weights).

## Vertex Model

Each vertex carries two scalars:

| Field | Type | Purpose |
|---|---|---|
| `vtx_state_[v]` | `float` | Internal state — updated by tanh(alpha * weighted_sum) with leak |
| `vtx_output_[v]` | `float` | Published output — copied from state after synchronous swap |

The separation of state and output is what makes synchronous update work: all vertices
read from the output array (last step's values) and write to the state array (this step's
values), then a single memcpy publishes the new states.

## Step: Gather and Activate

Each call to `Step()` executes two phases:

### Phase I: Compute New States

For each vertex v:
```
s = sum over neighbors i: vtx_output_[v ^ mask[i]] * weight[v][i]
raw = tanh(alpha * s)
vtx_state_[v] = retain * vtx_state_[v] + leak * raw
```

Where `retain = 1 - leak`. At leak=1.0 (the universal default), state is fully replaced
each step. At leak<1.0, old state is blended with the new activation (temporal smoothing).
Leak sweep experiments confirmed leak=1.0 is optimal across DIM 5-10.

### Phase II: Synchronous Swap

```
memcpy(vtx_output_, vtx_state_, N * sizeof(float))
```

All vertices update simultaneously. The output array is read-only during Phase I.

## Input Injection

Input is injected via `InjectInput()` before `Step()`. Two modes:

- **Single-input:** `InjectInput(float)` — one scalar, projected to all N vertices via W_in
- **Multi-input:** `InjectInput(float*)` — K scalars, projected via [N x K] W_in matrix

W_in weights are random uniform scaled by `input_scaling`. All N vertices receive input.
Input values are clamped to [-1, 1].

## Parameters

All parameters have per-DIM optimized defaults. Pass 0 for SR or input_scaling to use
the auto-tuned value.

| Parameter | Default | Role |
|---|---|---|
| `rng_seed` | — | Deterministic initialization seed |
| `alpha` | 1.0 | Tanh steepness — universally optimal |
| `leak` | 1.0 | Leak rate: 1.0 = no temporal smoothing (universally optimal) |
| `spectral_radius` | per-DIM | Target spectral radius (0 = auto) |
| `input_scaling` | per-DIM | W_in weight magnitude (0 = auto) |
| `num_inputs` | 1 | Number of input channels |

### Per-DIM Defaults

Jointly optimized on MG h=1 with full translation layer, LinearReadout, 3-seed average.
Stored in `Reservoir::DefaultSpectralRadius()` and `Reservoir::DefaultInputScaling()`.

| DIM | N    | SR   | input_scaling |
|-----|------|------|---------------|
| 5   | 32   | 0.90 | 1.50          |
| 6   | 64   | 1.00 | 1.00          |
| 7   | 128  | 1.05 | 0.80          |
| 8   | 256  | 1.10 | 0.50          |
| 9   | 512  | 1.00 | 0.40          |
| 10  | 1024 | 1.05 | 0.40          |

Pattern: higher DIM tolerates higher SR; smaller reservoirs need stronger input injection.

## Spectral Radius

Recurrent weights are initialized from uniform[-1,1] / sqrt(num_connections), then
rescaled via power iteration (up to 100 iterations with convergence check) to achieve
the target spectral radius. The spectral radius controls the reservoir's dynamical
regime: too low and it forgets quickly; too high and it becomes unstable.

## Computational Properties

- **O(N * DIM) per step** — each vertex sums 2*DIM weighted neighbor outputs
- **O(N²) total for dense ESN** — the hypercube achieves the same quality at O(N * DIM)
- **Zero adjacency storage** — neighbors computed by XOR
- **Trivially parallelizable** — all gathers are reads from the output array
- **Cache-friendly** — XOR addressing produces structured access patterns

See docs/Comparison.md for measured performance and timing vs a random sparse ESN
(equivalent results, equivalent speed at DIM 5-10).
