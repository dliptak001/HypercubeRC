# MemoryCapacityProfile Results

## What is Memory Capacity?

Memory Capacity (MC) measures how well a reservoir can reconstruct past inputs from its
current state. It was introduced by Jaeger (2001) as the fundamental metric for echo-state
networks. For each lag L, a linear readout is trained to predict input(t-L) from the
reservoir state at time t. The R² of this prediction tells you how much of the input
from L steps ago is still recoverable.

Total MC is the sum of R² across all lags. A perfect linear reservoir with N neurons
has a theoretical maximum MC of N. In practice, the tanh nonlinearity encodes input
history in a form that a linear readout cannot fully decode, so measured MC is always
well below N. This is not a deficiency — the same nonlinear encoding that limits linear
MC is what enables the reservoir to solve nonlinear tasks like NARMA-10 and Mackey-Glass.

The fading memory profile (R² vs lag) reveals the reservoir's temporal horizon: how far
back in time it can "see." A healthy reservoir shows smooth monotonic decay from R²~1
at short lags to R²~0 at long lags.

---

Run: 2026-03-23 | LinearReadout | Full translation (2.5N) | 3-seed avg {42,1042,2042}
Alpha: 1.0 | Leak: 1.0 | SR: per-DIM default | Input scaling: per-DIM default
Warmup: 200 (DIM 5-7) / 500 (DIM 8-10) | Collect: 18*N | MC = sum R² lags 1-50

## Per-Lag R² Profile

| DIM | N    | lag1   | lag2   | lag4   | lag8   | lag16  | lag32  | lag48  | MC   |
|-----|------|--------|--------|--------|--------|--------|--------|--------|------|
| 5   | 32   | 0.9920 | 0.9765 | 0.8860 | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 6.4  |
| 6   | 64   | 0.9967 | 0.9935 | 0.9785 | 0.8307 | 0.0000 | 0.0000 | 0.0000 | 10.1 |
| 7   | 128  | 0.9960 | 0.9919 | 0.9781 | 0.8994 | 0.2463 | 0.0000 | 0.0000 | 12.9 |
| 8   | 256  | 0.9982 | 0.9962 | 0.9904 | 0.9629 | 0.7910 | 0.0000 | 0.0000 | 19.8 |
| 9   | 512  | 0.9996 | 0.9992 | 0.9977 | 0.9902 | 0.9433 | 0.2032 | 0.0000 | 26.1 |
| 10  | 1024 | 0.9998 | 0.9995 | 0.9984 | 0.9933 | 0.9632 | 0.4671 | 0.0031 | 30.5 |

## Findings

- **Textbook fading memory curve.** Every DIM shows smooth monotonic decay from near-perfect
  R² at short lags to zero at long lags. No anomalies or oscillations.
- **Memory horizon extends with DIM.** The lag at which R² drops below 0.5 increases
  steadily: ~lag 5 (DIM=5), ~lag 10 (DIM=7), ~lag 20 (DIM=9), ~lag 32 (DIM=10).
- **MC scales with DIM.** From 6.4 (DIM=5) to 30.5 (DIM=10). Roughly linear growth,
  consistent with the theoretical expectation that MC scales with reservoir capacity.
- **Full translation vs raw comparison.** MC with translation is slightly lower than
  the main benchmark's raw MC (30.5 vs 33.2 at DIM=10). This is expected: MC is a
  linear reconstruction task, and the quadratic translation features (x², x*x') add
  dimensions without useful linear information for this task, slightly diluting the
  LinearReadout fit.
- **Short-lag R² approaches 1.0 at all DIM.** The reservoir perfectly tracks recent
  inputs. The bottleneck is long-lag recall, which improves with reservoir size.
