# Fast NARMA-K Diagnostic

## Why Not Just NARMA-10?

NARMA-10 is the standard RC benchmark for combined memory + nonlinear
computation, but it has two limitations for multi-timescale development:

1. **Fixed 10-step horizon.** The recurrence depends on inputs and outputs
   up to 10 steps back. A slow zone with leak=0.2 might remember 30+ steps,
   but NARMA-10 can't tell — it only needs 10. We need to dial K up to see
   where extended memory actually helps.

2. **Heavyweight benchmark.** The full NARMA-10 diagnostic runs 3 seeds,
   warmup + 18*N collect steps, both raw and translation features. That's
   fine for final reporting but too slow for sweeping leak rates and SR
   across DIMs.

## NARMA-K: The Generalization

The standard NARMA-10 recurrence (`SignalGenerators.h:120-121`):

```
y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1
```

Generalized to order K:

```
y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..K-1) + 1.5*u(t-(K-1))*u(t) + 0.1
```

Three things change when K increases:

- The **moving-average sum** extends over K terms instead of 10.
- The **delayed input product** `u(t-(K-1))*u(t)` reaches further back.
- The task requires the reservoir to **remember K steps** to reconstruct
  the target — directly stressing the memory horizon.

At K=10, this is exactly NARMA-10. At K=20 or K=30, it probes whether the
slow zone's extended memory translates into better nonlinear recall.

### Stability Note

The NARMA recurrence can become unstable at high K because the sum of K
terms grows with K. The existing implementation clamps `y(t)` to `[0, 1]`
(`SignalGenerators.h:122-123`), which prevents divergence. The coefficient
`0.05` on the sum term also helps — even at K=30, `0.05 * y(t) * sum` is
bounded by `0.05 * 1.0 * 30 = 1.5`. Combined with clamping, this should
remain stable up to at least K=50. Verify empirically at each new K.

## Fast Mode: What to Cut

For parameter sweeps (not final benchmarks), the diagnostic should run
lean:

| Setting | Full Benchmark | Fast Sweep |
|---------|---------------|------------|
| Seeds | 3 (42, 1042, 2042) | 1 (42) |
| Collect steps | 18*N | 8*N |
| Features | Raw + Translation | Raw only |
| Readout | Configurable | Ridge only |
| Output | Formatted table | Single NRMSE value |

This cuts wall time by roughly 6x (1 seed vs 3, half the steps, one
feature mode instead of two).

## What It Measures

Run NARMA-K at several values of K for a given reservoir configuration.
The result is an **NRMSE-vs-K curve**:

```
K=10:  NRMSE 0.35  ████████████
K=15:  NRMSE 0.42  ██████████████
K=20:  NRMSE 0.58  ███████████████████
K=25:  NRMSE 0.71  ████████████████████████
K=30:  NRMSE 0.89  ██████████████████████████████
```

With multi-timescale enabled (leak_slow < 1.0), the curve should decay
more slowly — the slow zone extends the usable memory horizon:

```
K=10:  NRMSE 0.34  ████████████          (similar to baseline)
K=15:  NRMSE 0.38  █████████████         (better)
K=20:  NRMSE 0.45  ███████████████       (much better)
K=25:  NRMSE 0.55  ██████████████████    (still usable)
K=30:  NRMSE 0.68  ██████████████████████ (better than baseline 0.89)
```

The **crossover point** — the K where multi-timescale starts winning — is
the key metric. It tells you the memory horizon where the slow zone begins
contributing.

## Implementation Notes

### Signal Generator

Generalize `GenerateNARMA10` (`SignalGenerators.h:106`) into
`GenerateNARMA(K, seed, steps)`. The existing function has K=10 hardcoded
in two places:

- The sum loop: `for (size_t i = 0; i < 10; ++i)` — replace with K.
- The delayed input: `u[t - 9]` — replace with `u[t - (K-1)]`.
- The history guard: `for (size_t t = 10; ...)` — replace with K.

The original `GenerateNARMA10` can become a thin wrapper:
`return GenerateNARMA(10, seed, steps);`

### Diagnostic Class

Model after `NARMA10` (`diagnostics/NARMA10.h`) but simplified:

- Template on DIM (same as existing).
- Constructor takes K and optional fast-mode flag.
- `Run()` returns a single NRMSE (no raw-vs-translation comparison in
  fast mode).
- `RunSweep(K_min, K_max, K_step)` runs multiple K values and returns
  the NRMSE-vs-K curve.

### Usage for Multi-Timescale Sweeps

```cpp
// Compare baseline vs multi-timescale at DIM=7
FastNarma<7> probe_baseline(/*K=*/20);       // leak=1.0 (default)
FastNarma<7> probe_multiscale(/*K=*/20);     // leak_slow=0.3

auto nrmse_base = probe_baseline.Run();
auto nrmse_multi = probe_multiscale.Run();
// If nrmse_multi < nrmse_base, the slow zone is helping at K=20
```
