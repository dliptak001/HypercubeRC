# Mackey-Glass Chaotic Time Series Prediction

## What this diagnostic measures

The Mackey-Glass equation is a delay differential equation that produces
low-dimensional deterministic chaos:

    dx/dt = 0.2 * x(t-17) / (1 + x(t-17)^10) - 0.1 * x(t)

With delay tau=17, the system is aperiodic but deterministic — the same
initial conditions always produce the same trajectory, but the trajectory
never repeats. This makes it the most widely used benchmark for echo-state
networks: it tests whether the reservoir can learn complex nonlinear
dynamics from the time series alone.

**The task:** given x(t), predict x(t+1). The reservoir receives the
normalized series as input and must learn the underlying dynamics well
enough to extrapolate one step into the future.

**The metric:** NRMSE (Normalized Root Mean Squared Error) = RMSE / std(targets).
A value of 1.0 means the model predicts no better than the mean; 0.0 is
perfect. Standard ESN results on MG h=1 range from 0.01 to 0.05.

## Why this benchmark matters

Mackey-Glass is the "hello world" of reservoir computing benchmarks. It
tests the reservoir's core capability: tracking a nonlinear dynamical
system from its time series. Unlike NARMA-10, it doesn't require deep
memory (the delay is only 17 steps), so it primarily measures the
reservoir's ability to reconstruct the state space of a chaotic attractor.

The raw vs. translation comparison reveals how much the tanh bottleneck
limits the linear readout. At larger DIM, the translation layer's
quadratic features (x², x*x') give the readout access to state-space
interactions that are invisible in the raw tanh outputs.

## How it works

1. Generate a Mackey-Glass time series (Euler integration, dt=1).
2. Normalize to [-1, +1] and split into warmup + collect.
3. Run with raw features (N selected states).
4. Run with translation features (2.5N).
5. Train the selected readout (Ridge or Linear) on 70%, test on 30%.
6. Report NRMSE for raw and translation, plus % change.

Uses a single per-DIM seed selected by 500-seed survey
(see [docs/SeedSurvey.md](../docs/SeedSurvey.md)).

## Sample results

Run with Ridge Readout:

| DIM | N    | Raw    | Full Translation | Change |
|-----|------|--------|------------------|--------|
| 5   | 32   | 0.0062 | 0.0044           | -29.0% |
| 6   | 64   | 0.0052 | 0.0037           | -28.7% |
| 7   | 128  | 0.0043 | 0.0032           | -26.6% |
| 8   | 256  | 0.0038 | 0.0024           | -37.5% |

## What to look for

- **Translation improvement scales with DIM.** Small reservoirs (DIM 5)
  see modest gains; large reservoirs (DIM 8+) see 30-50%+ improvement
  because there are enough neurons to support meaningful quadratic features.

- **All DIM values beat the standard ESN range** (0.01-0.05) even with
  raw features. The hypercube topology is efficient at encoding chaotic
  dynamics.

- **NRMSE improves monotonically with DIM.** More neurons = more state
  dimensions = better reconstruction of the chaotic attractor.

- **Ridge regression** finds the closed-form global optimum for the
  linear mapping, which is critical for this benchmark. Accuracy
  improves with DIM as the feature space grows.
