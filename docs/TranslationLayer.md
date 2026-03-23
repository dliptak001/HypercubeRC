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

## Implementation

The translation layer is implemented as two free functions in `TranslationLayer.h`:

- `TranslationTransform<DIM>(states, num_samples)` — applies the full x + x² + x*x'
  transform, returns a `std::vector<float>` of size num_samples * 2.5N.
- `TranslationFeatureCount<DIM>()` — returns 2.5N (constexpr).

Both are header-only templates, instantiated at compile time for DIM 5-10.
