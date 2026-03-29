# Translation Layer

## Purpose

The reservoir encodes input history into N vertex states via tanh activation. This
nonlinear encoding is powerful — it enables NARMA-10 and Mackey-Glass prediction — but
a linear readout cannot fully decode it. The tanh "folds" information into nonlinear
combinations that a linear model cannot untangle.

The translation layer bridges this gap by expanding N raw states into 2.5N features that
include quadratic terms the linear readout can directly use. This is analogous to a
polynomial kernel in an SVM: rather than changing the model, we change the features.

## Pipeline Position

    Reservoir (N states) -> TranslationLayer (2.5N features) -> Readout

The translation layer sits between the reservoir and the readout. It is a pure function
with no learnable parameters — it transforms each state snapshot independently using
fixed algebraic operations.

## Feature Classes

| Class | Count | Formula | Description |
|-------|-------|---------|-------------|
| x     | N     | state[v] | Raw vertex states (identity pass-through) |
| x²    | N     | state[v]² | Element-wise squared states |
| x*x'  | N/2   | state[v] * state[v XOR (N-1)] | Antipodal products |

**Total: N + N + N/2 = 2.5N features per timestep.**

### Why these three?

- **x (identity):** Preserves the raw linear information. Without this, the readout
  would lose access to the direct state values that dominate short-lag memory recall.

- **x² (squared):** Breaks the symmetry of tanh. The raw states are centered around zero
  (tanh output), so x² captures magnitude information that the linear readout otherwise
  misses. It also provides second-order self-interaction terms.

- **x*x' (antipodal products):** Cross-terms between maximally distant vertices on the
  hypercube. Vertex v's antipode is v XOR (N-1) — the bitwise complement, sitting at
  Hamming distance DIM. These products give the readout access to correlations between
  the most geometrically separated parts of the reservoir state.

### Output Layout

Per sample: `[x_0, x_1, ..., x_{N-1}, x_0², x_1², ..., x_{N-1}², x_0*x_0', x_1*x_1', ..., x_{N/2-1}*x_{N/2-1}']`

Only the first N/2 vertices generate antipodal products (v < N/2), since each pair
(v, v XOR (N-1)) is counted once.

## Feature Standardization

The three feature classes have different scale distributions:
- x: approximately uniform in [-1, 1] (tanh output)
- x²: biased positive, in [0, 1]
- x*x': approximately uniform in [-1, 1]

Both LinearReadout and RidgeRegression standardize features internally (zero mean, unit
variance) before training. This is critical: without standardization, the readout's
regularization term would penalize x² features less than x features (smaller magnitude),
leading to biased weight allocation.

## Computational Cost

The translation is O(N) per timestep — three passes over the N states. This is negligible
compared to the reservoir step cost (O(N * DIM) for the weighted sum + tanh).

Memory cost is 2.5N floats per collected timestep. For DIM=10 (N=1024), that's 2560
features * 18432 samples * 4 bytes ≈ 180 MB for a full training run. This is the
dominant memory cost in the pipeline.

## Does Antipodal Pairing Matter?

The comparison experiment (docs/Comparison.md) tested whether the specific geometric
pairing — antipodal vertices at maximum Hamming distance — provides an advantage over
random pairing. The result: **no measurable difference.** Any random pairing of vertices
for cross-terms produces equivalent results.

The antipodal pairing is retained for its computational elegance (v XOR (N-1) is a single
instruction) and its natural geometric interpretation on the hypercube, not because it
provides a unique dynamical advantage.

## Benchmark Results

Results from the full benchmark suite (`main.cpp`), 3-seed average {42, 1042, 2042},
LinearReadout, per-DIM optimized defaults for each mode.

### Mackey-Glass h=1 (NRMSE, lower is better)

| DIM |    N | Raw    | Translation | Change  |
|-----|------|--------|-------------|---------|
|   5 |   32 | 0.0097 | 0.0103      |  +6.2%  |
|   6 |   64 | 0.0085 | 0.0074      | -12.6%  |
|   7 |  128 | 0.0061 | 0.0053      | -13.5%  |
|   8 |  256 | 0.0068 | 0.0056      | -17.6%  |
|   9 |  512 | 0.0048 | 0.0040      | -17.6%  |
|  10 | 1024 | 0.0043 | 0.0032      | -25.2%  |

### NARMA-10 (NRMSE, lower is better)

| DIM |    N | Raw   | Translation | Change  |
|-----|------|-------|-------------|---------|
|   5 |   32 | 0.546 | 0.532       |  -2.6%  |
|   6 |   64 | 0.420 | 0.270       | -35.7%  |
|   7 |  128 | 0.395 | 0.191       | -51.5%  |
|   8 |  256 | 0.403 | 0.134       | -66.8%  |
|   9 |  512 | 0.388 | 0.080       | -79.4%  |
|  10 | 1024 | 0.385 | 0.074       | -80.8%  |

### Memory Capacity (sum R² lags 1-50, raw features only)

| DIM |    N |   MC |
|-----|------|------|
|   5 |   32 | 13.0 |
|   6 |   64 | 16.7 |
|   7 |  128 | 24.7 |
|   8 |  256 | 26.5 |
|   9 |  512 | 33.6 |
|  10 | 1024 | 32.9 |

### Assessment

**Mackey-Glass:** Translation improves prediction from DIM 6 onward, scaling from -12.6%
to -25.2% NRMSE reduction. DIM 5 is the exception (+6.2% worse) — with only 32 neurons
producing 80 translation features, the 2.5x expansion likely overfits given the small
state space. From DIM 6 upward, the benefit grows monotonically with reservoir size.

**NARMA-10:** The translation layer's strongest result. The improvement widens from -2.6%
at DIM 5 to -80.8% at DIM 10. NARMA-10's target equation contains explicit product terms
(y*sum, u*u) that align directly with the x² and x*x' features — the readout can learn
the nonlinear target almost directly from the expanded feature set instead of approximating
it linearly from raw tanh states. The 0.074 NRMSE at DIM 10 is well below the standard
ESN literature range of 0.2-0.4.

**Memory Capacity:** Reported for raw features only (the standard metric). MC increases
monotonically with N as expected, with a slight dip at DIM 10 (32.9 vs 33.6 at DIM 9).

**Overall:** The translation layer provides substantial gains on nonlinear tasks, with the
benefit increasing as reservoir size grows. The cost is modest — O(N) computation and 2.5x
memory — making it a strong default for tasks beyond simple linear recall.

## Implementation

The translation layer is implemented as two free functions in `TranslationLayer.h`:

- `TranslationTransform<DIM>(states, num_samples)` — applies the full x + x² + x*x'
  transform, returns a `std::vector<float>` of size num_samples * 2.5N.
- `TranslationFeatureCount<DIM>()` — returns 2.5N (constexpr).

Both are header-only templates, instantiated at compile time for DIM 5-10.
