# StateRank Results

## What is State Rank Analysis?

A reservoir computer's power comes from projecting input into a high-dimensional state
space. But how many of those dimensions are actually useful? State Rank Analysis answers
this by examining the eigenvalue spectrum of the state covariance matrix (X'X).

If the top few eigenvalues dominate (steep spectrum), the reservoir is effectively
low-dimensional — many vertices are redundant. If the spectrum is flat, the reservoir
uses its full capacity. The "effective rank" counts how many eigenvalues exceed 1% of the
largest, giving a single number for the reservoir's usable dimensionality.

The second part — input correlation analysis — asks a different question: is each vertex
responding to input, or generating autonomous noise? For each vertex, we compute R²
against 64 lagged input values. High R² means the vertex is encoding input history;
low R² means it's producing dynamics unrelated to the input. A healthy reservoir should
have high input correlation across all vertices — autonomous noise wastes capacity.

This diagnostic uses raw reservoir states (not translated features) because it analyzes
the intrinsic reservoir dynamics, not the prediction pipeline.

---

Run: 2026-03-23 | Raw features (intrinsic reservoir analysis) | 3-seed avg {42,1042,2042}
Alpha: 1.0 | Leak: 1.0 | SR: per-DIM default | Input scaling: per-DIM default
Warmup: 200 (DIM 5-7) / 500 (DIM 8-10) | Collect: 18*N | Max components: 30

## Eigenvalue Spectrum Summary

| DIM | N    | EV1    | EV2/EV1 | EV5/EV1 | EV10/EV1 | Eff Rank | Top-10 cumul |
|-----|------|--------|---------|---------|----------|----------|-------------|
| 5   | 32   | 3.79   | 40.7%   | 5.3%    | 0.5%     | 7        | 99.0%       |
| 6   | 64   | 3.29   | 58.2%   | 10.1%   | 0.8%     | 9        | 98.9%       |
| 7   | 128  | 6.63   | 65.0%   | 19.2%   | 3.3%     | 13       | 96.0%       |
| 8   | 256  | 5.89   | 81.4%   | 38.0%   | 12.9%    | 21       | 89.2%       |
| 9   | 512  | 7.23   | 81.5%   | 37.2%   | 10.7%    | 20       | 91.7%       |
| 10  | 1024 | 14.56  | 79.8%   | 40.5%   | 14.6%    | 23       | 87.3%       |

## Input Correlation Summary

| DIM | N    | Input-correlated | Mean R² | Min R²  | Max R²  | R²>0.5  |
|-----|------|-----------------|---------|---------|---------|---------|
| 5   | 32   | 99.8%           | 0.994   | 0.878   | 1.000   | 100.0%  |
| 6   | 64   | 99.4%           | 0.994   | 0.961   | 1.000   | 100.0%  |
| 7   | 128  | 96.8%           | 0.963   | 0.833   | 1.000   | 100.0%  |
| 8   | 256  | 95.2%           | 0.949   | 0.841   | 1.000   | 100.0%  |
| 9   | 512  | 98.6%           | 0.985   | 0.910   | 1.000   | 100.0%  |
| 10  | 1024 | 97.4%           | 0.972   | 0.879   | 1.000   | 100.0%  |

## Findings

- **Effective rank grows with DIM.** From 7 (DIM=5) to 23 (DIM=10). The eigenvalue
  spectrum flattens as DIM increases — the top eigenvalue captures 55% of variance at
  DIM=5 but only 20% at DIM=10. Larger reservoirs use more of their state dimensions.
- **The state space is genuinely high-dimensional.** At DIM=10, the top 10 eigenvalues
  capture only 87% of variance, meaning 13% is spread across the remaining ~1000
  dimensions. This is not a low-rank system.
- **95-100% of state variance is input-correlated.** Every vertex tracks input dynamics.
  There are no "dead" or autonomous-noise dimensions. The reservoir is fully engaged
  with the input stream.
- **100% of vertices have R² > 0.5.** Even the worst vertex (min R² = 0.83-0.96
  depending on DIM) is strongly correlated with the 64-lag input history. The hypercube
  topology distributes input information uniformly across all vertices.
- **No structural symmetry collapse.** Despite the hypercube's high symmetry, the
  reservoir does not produce redundant state dimensions. The combination of random
  weights, two connection types (shells + nearest), and tanh nonlinearity breaks the
  graph symmetry effectively.
