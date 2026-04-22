# State Rank Analysis

## What this diagnostic measures

A reservoir computer's power comes from projecting a 1-dimensional input
into an N-dimensional state space. But how many of those N dimensions are
actually useful? State Rank Analysis answers this with two complementary
metrics.

### 1. Eigenvalue spectrum — how many dimensions does the reservoir use?

The state covariance matrix (X'X) captures how the N-dimensional reservoir
states are distributed. Its eigenvalues reveal the shape of that distribution:

- **Steep spectrum** (top eigenvalue dominates): the reservoir is
  effectively low-dimensional. Many neurons are producing redundant
  outputs, and the state space collapses to a lower-dimensional manifold.

- **Flat spectrum** (eigenvalues roughly equal): the reservoir uses its
  full capacity. Each neuron contributes unique information, and the
  state space fills all N dimensions.

The **effective rank** counts how many eigenvalues exceed 1% of the
largest, giving a single number for the reservoir's usable dimensionality.

### 2. Input correlation — is each vertex driven by input or by noise?

For each vertex, we compute R² against 64 lagged input values. This
answers a different question than the eigenvalue spectrum:

- **High R²** means the vertex's output is explainable from the input
  history. It is encoding useful information.

- **Low R²** means the vertex is producing autonomous dynamics unrelated
  to the input. This wastes capacity on noise.

A healthy reservoir should have high input correlation across all vertices.
The hypercube topology's uniform connectivity is designed to distribute
input information to every vertex.

This diagnostic uses **raw reservoir states** (not translated features)
because it analyzes intrinsic reservoir properties, not prediction quality.

## Why this diagnostic matters

State rank tells you whether the reservoir is using its neurons efficiently:

- **Too few effective dimensions** means the reservoir is redundant —
  you could get the same performance with a smaller one.

- **Low input correlation** means some neurons are "dead weight" —
  producing outputs that don't relate to the input stream.

- **Symmetry collapse** would appear as a low effective rank despite
  many neurons. The hypercube's high structural symmetry could
  theoretically cause this, but the combination of random weights, two
  connection families (shells + nearest-neighbor), and tanh nonlinearity
  should break the symmetry effectively. This diagnostic verifies that.

## How it works

1. Generate uniform random inputs in [-1, +1] (deterministic per seed).
2. Drive the reservoir and collect raw states.
3. Mean-center the state matrix.
4. Compute eigenvalues via deflated power iteration (up to 30 components).
5. For each vertex, compute R² against 64 lagged inputs.
6. Average across 3 seeds {42, 1042, 2042}.

## Sample results

Run with DIM=8, N=256, 3-seed average:

**Eigenvalue spectrum (top 10):**

| #  | Eigenvalue | % of max | Cumulative % |
|----|-----------|----------|--------------|
| 1  | 5.89      | 100.0%   | 11.2%        |
| 2  | 4.80      | 81.4%    | 20.3%        |
| 5  | 2.24      | 38.0%    | 49.1%        |
| 10 | 0.76      | 12.9%    | 89.2%        |

Effective rank (>1% of max): 21 of 30 computed.

**Input correlation:**

| Metric | Value |
|--------|-------|
| Input-correlated variance | 95.2% |
| Mean per-vertex R² | 0.949 |
| Min per-vertex R² | 0.841 |
| Vertices with R² > 0.5 | 100.0% |

## What to look for

- **Effective rank grows with DIM.** From ~7 at DIM=5 to ~23 at DIM=10.
  The spectrum flattens as DIM increases — larger reservoirs use more of
  their state dimensions.

- **The state space is genuinely high-dimensional.** At DIM=10, the top
  10 eigenvalues capture only ~87% of variance, meaning ~13% is spread
  across the remaining ~1000 dimensions.

- **95-100% of variance is input-correlated.** Every vertex tracks input
  dynamics. There are no "dead" dimensions producing autonomous noise.

- **100% of vertices have R² > 0.5.** Even the worst vertex is strongly
  correlated with input history. The hypercube topology distributes input
  information uniformly — no vertex is isolated.

- **No symmetry collapse.** Despite the hypercube's high structural
  symmetry, the reservoir produces non-redundant state dimensions. Random
  weights, dual connection families, and tanh nonlinearity break the
  graph symmetry effectively.
