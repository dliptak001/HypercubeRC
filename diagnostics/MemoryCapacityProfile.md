# Memory Capacity Profile

## What this diagnostic measures

Memory Capacity (MC) quantifies how well a reservoir can reconstruct past
inputs from its current state. Introduced by Jaeger (2001), it is the
fundamental metric for echo-state networks.

**The idea:** for each lag L from 1 to 50, train a linear readout to
predict the input from L steps ago using the current reservoir state.
The R² of each prediction tells you how much of that lagged input is
still recoverable. Total MC is the sum of R² across all lags.

**Theoretical maximum:** a perfect linear reservoir with N neurons can
achieve MC = N. In practice, MC is always well below N because the tanh
nonlinearity encodes input history in a compressed, nonlinear form that
a linear readout cannot fully decode. This is not a deficiency — the same
nonlinear encoding that limits linear MC is what enables the reservoir to
solve nonlinear tasks like NARMA-10 and Mackey-Glass.

## Why this benchmark matters

The per-lag R² profile reveals the reservoir's **temporal horizon** — how
far back in time it can "see":

- **Short lags (1-4):** R² near 1.0 means the reservoir perfectly tracks
  recent inputs. Every healthy reservoir should score well here.

- **Medium lags (8-20):** This is where reservoir size matters. A DIM=5
  reservoir (32 neurons) loses most information by lag 8; a DIM=8
  reservoir (256 neurons) retains strong recall through lag 16.

- **Long lags (32-50):** Only the largest reservoirs retain useful
  information this far back. R² drops to zero when the fading memory
  has fully washed out.

A healthy reservoir shows smooth, monotonic decay from R²~1 at short lags
to R²~0 at long lags. Oscillations or plateaus would suggest structural
problems.

## How it works

1. Generate uniform random inputs in [-1, +1] (deterministic per seed).
2. Drive the reservoir and apply the full translation layer (2.5N features).
3. For each lag L from 1 to 50:
   - Build targets: target[t] = input[t - L].
   - Train the selected readout on 70%, compute R² on 30%.
   - If R² > 0, add it to the MC sum.
4. Report per-lag R² at selected display lags plus total MC.

Uses a single per-DIM seed selected by 500-seed survey
(see [docs/SeedSurvey.md](../docs/SeedSurvey.md)).

**Note:** This diagnostic uses full translation features (2.5N), which
differs from the main benchmark's MC (raw N features). The main.cpp
benchmark reports raw MC for comparability with published results.

## Sample results

Run with Ridge Readout, full translation, DIM=8:

| Lag | R² |
|-----|----|
| 1   | 0.9982 |
| 2   | 0.9962 |
| 4   | 0.9904 |
| 8   | 0.9629 |
| 16  | 0.7910 |
| 32  | 0.0000 |
| 48  | 0.0000 |

MC (all lags 1-50): ~19.8

## What to look for

- **Smooth monotonic decay.** Every DIM should show a clean sigmoid-like
  drop from R²~1 to R²~0. No oscillations or anomalies.

- **Memory horizon extends with DIM.** The lag at which R² drops below
  0.5 increases steadily with reservoir size — roughly lag 5 at DIM=5,
  lag 10 at DIM=7, lag 20 at DIM=9.

- **MC scales with DIM.** From ~6 (DIM=5) to ~30 (DIM=10). Roughly
  linear growth, consistent with the theoretical expectation that MC
  scales with reservoir capacity.

- **Short-lag R² near 1.0 at all DIM.** The reservoir perfectly tracks
  recent inputs regardless of size. The bottleneck is always long-lag
  recall.

- **Translation vs. raw MC.** MC with translation features can be
  slightly different from raw MC. The extra quadratic features don't
  add useful linear information for this task (memory recall is a linear
  reconstruction problem), but the Ridge readout handles the extra
  dimensions without overfitting.
