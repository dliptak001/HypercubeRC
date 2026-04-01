# Hypercube vs. Random Sparse ESN — Does Topology Matter?

## The question

A fundamental question for any structured reservoir: does the topology
actually matter, or would a random graph work just as well?

If a random sparse ESN with the same vertex count and connection density
produces equivalent results, then the hypercube's value is computational
(XOR addressing, zero adjacency storage) rather than dynamical. This
document presents a controlled head-to-head experiment to find out.

## Experimental setup

Both reservoirs share identical parameters — the only difference is how
neighbors are chosen:

| Property | Hypercube (`Reservoir<DIM>`) | Random (`RandomESN<DIM>`) |
|----------|---------------------------|--------------------------|
| Vertices | N = 2^DIM | N = 2^DIM |
| Degree | 2*DIM - 2 per vertex | 2*DIM - 2 per vertex |
| Weight init | uniform[-1,1] / sqrt(2*DIM - 2) | same |
| SR | 0.90 (scale-invariant) | same |
| Input scaling | 0.02 (scale-invariant) | same |
| Activation | tanh(alpha * sum), alpha=1.0 | same |
| W_in | random projection, same RNG | same |
| **Neighbors** | **XOR masks (shells + nearest)** | **2*DIM - 2 random distinct** |
| Adjacency storage | **none** (computed inline) | **N * (2*DIM - 2) array** |

The `RandomESN<DIM>` was a temporary class built for this experiment and
has since been removed from the codebase — the experiment answered its
question.

**Readout:** LinearReadout (SGD), raw features (no translation layer).
This isolates the topology's effect on the reservoir dynamics, without
the translation layer's nonlinear features masking subtle differences.

**Seeds:** 3-seed average {42, 1042, 2042}. The same seeds produce the
same weights in both topologies — only the connectivity pattern differs.

## Performance results

### Memory Capacity (sum of R², lags 1-50)

| DIM | N    | Hypercube | Random | Delta |
|-----|------|-----------|--------|-------|
| 5   | 32   | 6.7       | 6.7    | 0%    |
| 6   | 64   | 7.4       | 9.7    | +31%  |
| 7   | 128  | 21.5      | 14.1   | -34%  |
| 8   | 256  | 27.6      | 20.7   | -25%  |
| 9   | 512  | 28.5      | 30.6   | +7%   |
| 10  | 1024 | 33.2      | 37.1   | +12%  |

No consistent winner. Hypercube leads at DIM 7-8; Random leads at DIM 6, 9-10.

### Mackey-Glass h=1 (NRMSE, lower is better)

| DIM | N    | Hypercube | Random | Delta |
|-----|------|-----------|--------|-------|
| 5   | 32   | 0.029     | 0.018  | -38%  |
| 6   | 64   | 0.019     | 0.016  | -18%  |
| 7   | 128  | 0.012     | 0.010  | -19%  |
| 8   | 256  | 0.009     | 0.014  | +60%  |
| 9   | 512  | 0.004     | 0.004  | -10%  |
| 10  | 1024 | 0.003     | 0.002  | -27%  |

Mixed results. Both converge to excellent performance at DIM 9-10.

### NARMA-10 (NRMSE, lower is better)

| DIM | N    | Hypercube | Random | Delta |
|-----|------|-----------|--------|-------|
| 5   | 32   | 0.829     | 0.805  | -3%   |
| 6   | 64   | 0.719     | 0.658  | -8%   |
| 7   | 128  | 0.451     | 0.492  | +9%   |
| 8   | 256  | 0.407     | 0.493  | +21%  |
| 9   | 512  | 0.411     | 0.424  | +3%   |
| 10  | 1024 | 0.406     | 0.414  | +2%   |

Hypercube has a modest advantage at DIM 7-8; both converge at DIM 9-10.

## Computational comparison

Step timing: InjectInput + Step, best of 3 runs, Release build (-O3
-march=native -ffast-math), MinGW g++, single-threaded.

| DIM | N    | Hypercube (us/step) | Random (us/step) | Diff |
|-----|------|---------------------|-------------------|------|
| 5   | 32   | 8.54                | 8.64              | +1%  |
| 6   | 64   | 9.03                | 8.87              | -2%  |
| 7   | 128  | 10.14               | 10.10             | 0%   |
| 8   | 256  | 12.23               | 12.56             | +3%  |
| 9   | 512  | 17.06               | 17.40             | +2%  |
| 10  | 1024 | 27.50               | 27.02             | -2%  |

All within measurement noise. No measurable speed difference at these sizes.

### Why no speed difference?

The hypercube's theoretical advantage is O(1) XOR addressing vs. stored
adjacency lookup. In practice at DIM 5-12:

- **Adjacency fits in cache.** At DIM=10 the adjacency array is
  1024 * 20 * 4B = 80KB — comfortably in L2. Sequential vertex-by-vertex
  access lets the prefetcher stay ahead.

- **The bottleneck is computation, not addressing.** Each step performs
  N * (2*DIM - 2) multiply-accumulates plus N tanh evaluations. The XOR vs.
  array-index for neighbor lookup is a negligible fraction of total work.

- **Both scale as O(N * DIM).** Neither is dense (O(N²)). The constant
  factor difference between a XOR instruction and an array load is
  invisible at these sizes.

The XOR advantage would become meaningful at much larger N where adjacency
storage pressures cache capacity, or in hardware implementations (FPGA,
neuromorphic) where memory bandwidth is the bottleneck.

## What the results tell us

### Topology does not determine reservoir quality

Across three standard benchmarks and six reservoir sizes, neither topology
shows a consistent advantage. The differences are within the range of
seed-to-seed variance. A random sparse graph with the same degree produces
equivalent dynamical richness to the structured hypercube.

This is consistent with the broader reservoir computing literature: what
matters is the spectral radius, connection density, and nonlinearity — not
the specific graph topology. The reservoir's job is to create a
high-dimensional, fading-memory embedding of the input stream, and both
topologies accomplish this equally well.

### The hypercube's value is architectural, not dynamical

The hypercube does not compute better — but it computes more elegantly:

- **Zero storage overhead.** No adjacency list. The reservoir needs only
  two arrays (states, weights); a random graph adds a third (adjacency).
  At DIM=10 this is 80KB — trivial now, but it scales as O(N * DIM)
  additional memory.

- **Implicit structure.** XOR addressing is deterministic and reproducible
  without storing the graph. Two implementations with the same DIM
  automatically agree on connectivity — no serialization needed.

- **Parallelization.** XOR-based neighbor computation is trivially
  SIMD-able with no data dependencies beyond the read-only output array.

- **Hardware mapping.** In FPGA or neuromorphic implementations, XOR
  addressing maps to gates directly — no routing table, no memory
  controller for adjacency lookup.

None of these advantages appear in a software benchmark at DIM 5-12 on a
modern CPU with deep cache hierarchies. They become relevant at scale or
in constrained environments.

### Note on readout type

This experiment used LinearReadout (SGD) with raw features to isolate the
topology's effect. The project now also supports Ridge regression (see
`ReadoutType` in `ESN.h`), which provides the closed-form optimal readout.
Ridge regression generally improves absolute NRMSE numbers (particularly
for NARMA-10 and MG at higher DIM), but since the readout is independent
of the reservoir topology, the relative comparison between hypercube and
random would remain the same — both topologies produce equally rich state
spaces for the readout to work with.

### Implications for the translation layer

The translation layer's antipodal products (x * x', where x' = state[v XOR
(N-1)]) pair each vertex with the vertex at the opposite corner of the
hypercube. Since topology doesn't affect reservoir quality, this pairing
works for geometric convenience, not because antipodal vertices carry
complementary information. Any random pairing of vertices for cross-term
products would likely produce equivalent results.

The antipodal pairing is retained for its computational elegance — a single
XOR with the complement mask — and its natural geometric interpretation on
the hypercube.
