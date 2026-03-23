# Hypercube vs Random Sparse ESN Comparison

## Purpose

A fundamental question for any structured reservoir: does the topology actually matter,
or would a random graph work just as well? If a random sparse ESN with the same vertex
count and connection density produces equivalent results, then the hypercube's value is
purely computational (XOR addressing, zero storage) rather than dynamical.

This document presents head-to-head results from a controlled experiment using the
hypercube `Reservoir<DIM>` and a temporary `RandomESN<DIM>` baseline (since removed
from the codebase — the experiment answered its question).

## Experimental Setup

Both reservoirs share:
- Same N = 2^DIM vertices, same 2*DIM connections per vertex
- Same weight initialization: uniform[-1,1] / sqrt(2*DIM), rescaled to target SR
- Same per-DIM spectral radius and input scaling defaults
- Same tanh activation, leak=1.0, alpha=1.0
- Same W_in initialization (separate RNG for adjacency ensures identical W_rec and W_in
  for a given seed — only the connectivity pattern differs)
- LinearReadout, raw features (no translation layer), 3-seed average {42, 1042, 2042}
- collect = 18*N samples, 70/30 train/test split

The only difference: Reservoir computes neighbors via XOR masks (Hamming shells + nearest
neighbors); RandomESN draws 2*DIM distinct random neighbors per vertex and stores them
in an adjacency array.

## Performance Comparison

### Memory Capacity (sum of R², lags 1-50, LinearReadout)

| DIM | N    | Reservoir | RandomESN | Delta |
|-----|------|--------|-----------|-------|
| 5   | 32   | 6.7    | 6.7       | 0%    |
| 6   | 64   | 7.4    | 9.7       | +31%  |
| 7   | 128  | 21.5   | 14.1      | -34%  |
| 8   | 256  | 27.6   | 20.7      | -25%  |
| 9   | 512  | 28.5   | 30.6      | +7%   |
| 10  | 1024 | 33.2   | 37.1      | +12%  |

No consistent winner. Reservoir leads at DIM 7-8; RandomESN leads at DIM 6, 9-10.

### Mackey-Glass h=1 (NRMSE, lower is better, LinearReadout, raw)

| DIM | N    | Reservoir | RandomESN | Delta |
|-----|------|--------|-----------|-------|
| 5   | 32   | 0.029  | 0.018     | -38%  |
| 6   | 64   | 0.019  | 0.016     | -18%  |
| 7   | 128  | 0.012  | 0.010     | -19%  |
| 8   | 256  | 0.009  | 0.014     | +60%  |
| 9   | 512  | 0.004  | 0.004     | -10%  |
| 10  | 1024 | 0.003  | 0.002     | -27%  |

Mixed results. RandomESN wins at DIM 5-7 and 10; Reservoir wins at DIM 8. Both converge
to excellent performance at DIM 9-10.

### NARMA-10 (NRMSE, lower is better, LinearReadout, raw)

| DIM | N    | Reservoir | RandomESN | Delta |
|-----|------|--------|-----------|-------|
| 5   | 32   | 0.829  | 0.805     | -3%   |
| 6   | 64   | 0.719  | 0.658     | -8%   |
| 7   | 128  | 0.451  | 0.492     | +9%   |
| 8   | 256  | 0.407  | 0.493     | +21%  |
| 9   | 512  | 0.411  | 0.424     | +3%   |
| 10  | 1024 | 0.406  | 0.414     | +2%   |

Reservoir has a modest advantage at DIM 7-8; RandomESN wins at DIM 5-6. Both converge
at DIM 9-10.

## Computational Comparison

Step timing: InjectInput + Step, best of 3 runs, Release build (-O3 -march=native
-ffast-math), MinGW g++, single-threaded (N < 4096, OMP threshold not reached).

| DIM | N    | Reservoir (us/step) | RandomESN (us/step) | Diff |
|-----|------|------------------|---------------------|------|
| 5   | 32   | 8.54             | 8.64                | +1%  |
| 6   | 64   | 9.03             | 8.87                | -2%  |
| 7   | 128  | 10.14            | 10.10               | 0%   |
| 8   | 256  | 12.23            | 12.56               | +3%  |
| 9   | 512  | 17.06            | 17.40               | +2%  |
| 10  | 1024 | 27.50            | 27.02               | -2%  |

All within measurement noise. No measurable speed difference.

### Why no speed difference?

The hypercube's theoretical advantage is O(1) XOR addressing vs stored adjacency lookup.
In practice at DIM 5-10:

- **RandomESN adjacency fits in cache.** At DIM=10 the adjacency array is 1024 * 20 *
  4B = 80KB — comfortably in L2. The memory access pattern is sequential (vertex-by-vertex),
  so prefetching works well.
- **The bottleneck is computation, not addressing.** Each step performs N * 2*DIM
  multiply-accumulates plus N tanh evaluations. The XOR vs array-index for neighbor
  lookup is a negligible fraction of the total work.
- **Both scale as O(N * DIM).** Neither is dense (O(N²)). The constant factor difference
  between a XOR instruction and an array load is invisible at these sizes.

The XOR addressing advantage would become meaningful at much larger N where adjacency
storage pressures cache capacity, or in hardware implementations where memory bandwidth
is the bottleneck.

## Discussion

### Topology does not determine reservoir quality

Across three standard benchmarks (MC, MG, NARMA-10) and six reservoir sizes (DIM 5-10),
neither topology shows a consistent advantage. The differences are within the range of
seed-to-seed variance. A random sparse graph with the same degree produces equivalent
dynamical richness to the structured hypercube.

This is consistent with the broader reservoir computing literature: what matters is the
spectral radius, connection density, and nonlinearity — not the specific graph topology.
The reservoir's job is to create a high-dimensional, fading-memory embedding of the input
stream, and both topologies accomplish this equally well.

### The hypercube's value is architectural, not dynamical

The hypercube does not compute better — but it computes *more elegantly*:

- **Zero storage overhead.** No adjacency list needed. Reservoir has two arrays (states,
  weights); RandomESN needs a third (adjacency). At DIM=10 this is 80KB — trivial now,
  but it scales as O(N * DIM) additional memory.
- **Implicit structure.** The XOR addressing is deterministic and reproducible without
  storing the graph. Two implementations with the same DIM automatically agree on
  connectivity.
- **Parallelization simplicity.** XOR-based neighbor computation is trivially SIMD-able
  and has no data dependencies beyond the read-only output array.
- **Hardware mapping.** In FPGA or neuromorphic implementations, XOR addressing maps
  to gates directly — no routing table, no memory controller for adjacency lookup.

None of these advantages show up in a software benchmark at DIM 5-10 on a modern CPU
with deep cache hierarchies. They become relevant at scale or in constrained environments.

### Implications for the translation layer

Since topology doesn't affect reservoir quality, the antipodal pairing in the translation
layer (x * x', where x' = state[v XOR (N-1)]) works for geometric convenience, not
because antipodal vertices carry complementary information. Any random pairing of vertices
for cross-term products would produce equivalent results. The pairing is retained for its
computational elegance (a single XOR instruction) and natural geometric interpretation.
