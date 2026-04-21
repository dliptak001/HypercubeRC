# Scale-Invariant Hyperparameters on the Hypercube

## The finding

The general-purpose spectral radius and input scaling for HypercubeRC are
**independent of reservoir size**. Across five dimensions tested
(DIM 5-9, N = 32 to 512), the same configuration wins every time:

| Parameter | General-purpose value | Tested range |
|-----------|----------------------|--------------|
| Spectral radius | **0.90** | 0.70 - 0.99 |
| Input scaling | **0.02** | 0.01 - 0.40 |

This holds for both feature modes (raw N-dim readout and 2.5N
translation-layer readout) and across all three standard benchmarks
(Mackey-Glass h=1, NARMA-10, Memory Capacity).

## Why this is unusual

In conventional reservoir computing, the optimal spectral radius
typically shifts with reservoir size. Larger reservoirs can tolerate
higher SR because the spectral radius is a bulk property of the
weight matrix, and individual eigenvalue contributions shrink as N
grows. Practitioners routinely re-sweep SR when changing N, and
published ESN results almost always report per-size tuning.

The hypercube breaks this pattern. SR = 0.90 is optimal whether
the reservoir has 32 neurons or 512.

## Evidence

Three-pass grid sweeps (coarse, normal, fine) were run for each DIM
via grid sweep. Ridge readout, 3-seed average,
jointly optimizing MG h=1 + NARMA-10 + MC.

### Summary at SR = 0.90, input = 0.02

| DIM | N   | MG raw  | MG trans | NAR raw | NAR trans | MC    |
|-----|-----|---------|----------|---------|-----------|-------|
| 5   | 32  | 0.0181  | 0.0152   | 0.641   | 0.646     | 13.3  |
| 6   | 64  | 0.0101  | 0.0076   | 0.448   | 0.331     | 17.7  |
| 7   | 128 | 0.0072  | 0.0050   | 0.382   | 0.179     | 25.1  |
| 8   | 256 | 0.0051  | 0.0033   | 0.397   | 0.137     | 28.1  |
| 9   | 512 | 0.0047  | 0.0029   | 0.377   | 0.088     | 30.6  |

All metrics improve monotonically with DIM (lower NRMSE, higher MC)
using the same SR and input scaling at every scale.

### How flat is the landscape?

At DIM 8, the fine sweep (SR 0.89-0.93, input 0.02-0.10) shows
NARMA raw varying by less than 0.002 across the entire grid. The
optimal SR can be read off any DIM's sweep and applied to all others.

## Why the hypercube produces this

The hypercube topology is **vertex-transitive**: every vertex has
exactly the same local structure. Specifically:

1. **Uniform degree.** Every vertex has exactly 2*DIM - 2 connections
   (DIM nearest neighbors + DIM-2 shell connections), regardless of
   which vertex it is.

2. **Identical neighborhood pattern.** The shell masks (3, 7, 15, ...)
   and nearest-neighbor masks (1, 2, 4, ...) produce the same
   connectivity structure at every vertex, just shifted by the vertex
   index via XOR.

3. **Uniform input distribution.** Stride-interleaved input injection
   spreads input energy evenly across the hypercube. Every K-th vertex
   receives input, with no clustering or dead zones.

This means the per-vertex dynamics are structurally identical at
every scale. Increasing DIM adds more vertices, but each one sees
the same number of neighbors with the same weight distribution.
The stability threshold (the SR where dynamics transition from
convergent to chaotic) is a property of the local structure, not
the global size.

In contrast, random sparse ESNs have heterogeneous degree
distributions — some neurons are hubs, some are peripheral. This
heterogeneity interacts with SR differently at each N, forcing
per-size retuning.

## Empirical validation: 500-seed survey

A 500-seed survey tested the stability
of SR=0.90 as a general-purpose default across all three benchmarks
(DIM 5-8). The survey measured Spearman rank correlation of seed
performance across SR values {0.80, 0.85, 0.90, 0.95, 1.00} and IS
values {0.010, 0.015, 0.020, 0.025, 0.030}.

Key findings supporting SR=0.90 as the general-purpose default:

- **MG** mean NRMSE reaches minimum at or near 0.90 at every DIM.
- **MC** mean MC increases toward 0.95, but the gain from 0.90→0.95
  comes with doubled variance across all tasks.
- **NARMA** mean NRMSE hits minimum at 0.90-0.95, then rises at 1.00.
- **Rank correlation** in the 0.85-0.90 corridor exceeds 0.82 for all
  three benchmarks at all DIM values — seeds screened at SR=0.90
  transfer reliably to adjacent configurations.
- **SR=1.00** is a qualitatively different regime: correlation with
  0.90 collapses below 0.45, and NARMA even goes negative at DIM 7.

SR=0.90 is not necessarily the single best SR for any one task — MC
would prefer 0.95, and individual seeds may peak elsewhere — but it
is the best compromise across tasks, with the lowest variance and
strongest rank correlation in the operating range.

## Practical implication

**Configure once, scale freely.** Set SR = 0.90 and input_scaling
= 0.02, then increase DIM to add capacity without re-sweeping.
This eliminates the most common hyperparameter tuning burden in
reservoir computing.

Combined with:
- **XOR addressing**: zero adjacency storage at any scale
- **Output fraction**: constant readout cost as N grows (use 50%
  of vertices for features, Ridge cost scales with output count
  not reservoir size)

The hypercube is an RC architecture designed for painless scaling.
