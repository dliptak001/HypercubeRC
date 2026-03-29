# Translation Layer — Breaking Through the tanh Bottleneck

## The problem

Every reservoir neuron outputs tanh(sum), which compresses its internal
dynamics into the range [-1, +1]. A linear readout can only form weighted
sums of these outputs — it cannot "undo" the tanh to recover products or
squared terms that the reservoir computed internally.

This creates a bottleneck: the reservoir may be computing rich nonlinear
dynamics, but the linear readout can only see a compressed, folded version
of them. Information that the reservoir encodes as products or magnitudes
is invisible to a purely linear model.

## The solution

The translation layer creates those nonlinear features explicitly, giving
the readout direct access to information that would otherwise be trapped
inside tanh. This is analogous to a polynomial kernel in an SVM: rather
than making the model more complex, we expand the features.

```
Reservoir (N states) ──> Translation Layer (2.5N features) ──> Readout
```

The translation layer is a pure function with no learnable parameters.
It transforms each state snapshot independently using fixed algebraic
operations.

## Three feature classes

| Class | Count | Formula | What it captures |
|-------|-------|---------|-----------------|
| x     | N     | state[v] | Raw linear information — direct state values |
| x²    | N     | state[v]² | Magnitude regardless of sign — energy-like information |
| x*x'  | N/2   | state[v] * state[v XOR (N-1)] | Long-range correlations across the hypercube |

**Total: N + N + N/2 = 2.5N features per timestep.**

### Why each feature class helps

**x (identity):** Preserves the raw linear information. Without this,
the readout would lose access to the direct state values that dominate
short-lag memory recall. This is the baseline that the other two classes
build on.

**x² (squared):** Breaks the symmetry of tanh. Raw states are centered
around zero, so positive and negative activations of equal magnitude
cancel out in a linear sum. Squaring captures the magnitude regardless
of sign — how strongly each neuron is responding, not just in which
direction. It also provides second-order self-interaction terms that
help reconstruct nonlinear target functions.

**x*x' (antipodal products):** Cross-terms between maximally distant
vertices on the hypercube. Vertex v's antipode is v XOR (N-1) — the
bitwise complement, sitting at Hamming distance DIM (the maximum
possible distance). These products give the readout access to
correlations between the most geometrically separated parts of the
reservoir, capturing long-range dynamics that no single vertex encodes.

Only the first N/2 vertices generate antipodal products (v < N/2),
since each pair (v, v XOR (N-1)) is counted once.

### Output layout

Per sample:
```
[x_0, x_1, ..., x_{N-1}, x_0², x_1², ..., x_{N-1}², x_0*x_0', x_1*x_1', ..., x_{N/2-1}*x_{N/2-1}']
```

## Feature standardization

The three feature classes have different scale distributions:

| Class | Range | Distribution |
|-------|-------|-------------|
| x     | [-1, +1] | Approximately symmetric around zero |
| x²    | [0, 1] | Biased positive (always non-negative) |
| x*x'  | [-1, +1] | Approximately symmetric around zero |

Both LinearReadout and RidgeRegression standardize features internally
(zero mean, unit variance) before training. This is critical: without
standardization, the readout's regularization would penalize the
feature groups unevenly based on their raw scale, leading to biased
weight allocation. See the readout class briefs in `readout/LinearReadout.h`
and `readout/RidgeRegression.h` for details.

## Does antipodal pairing matter?

The comparison experiment (`docs/DoesTopologyMatter.md`) tested whether
the specific geometric pairing — antipodal vertices at maximum Hamming
distance — provides an advantage over random pairing. The result:
**no measurable difference.** Any random pairing of vertices for
cross-terms would likely produce equivalent results.

The antipodal pairing is retained for its computational elegance
(v XOR (N-1) is a single instruction) and its natural geometric
interpretation on the hypercube, not because it provides a unique
dynamical advantage.

## Impact on benchmark performance

The translation layer's impact scales with task difficulty. Results
from the main benchmark suite (Ridge Readout, 3-seed average):

### Mackey-Glass h=1 (NRMSE, lower is better)

| DIM | N    | Raw    | Translation | Change |
|-----|------|--------|-------------|--------|
| 5   | 32   | 0.0174 | 0.0141      | -18.8% |
| 6   | 64   | 0.0106 | 0.0074      | -29.7% |
| 7   | 128  | 0.0062 | 0.0045      | -28.1% |
| 8   | 256  | 0.0060 | 0.0039      | -35.1% |

### NARMA-10 (NRMSE, lower is better)

| DIM | N    | Raw   | Translation | Change |
|-----|------|-------|-------------|--------|
| 5   | 32   | 0.566 | 0.539       | -4.7%  |
| 6   | 64   | 0.417 | 0.264       | -36.7% |
| 7   | 128  | 0.387 | 0.176       | -54.6% |
| 8   | 256  | 0.399 | 0.125       | -68.6% |

### What the numbers show

**Mackey-Glass:** 19-35% NRMSE improvement at DIM 5-8. The chaotic
time series has smooth dynamics that a linear readout can partially
decode from raw states, so the translation layer provides a moderate
boost by exposing the quadratic interactions.

**NARMA-10:** This is where the translation layer earns its keep.
5-69% improvement, scaling dramatically with DIM. The NARMA-10 target
contains explicit product terms (y*sum, u*u) that align directly with
the x² and x*x' features — the readout can learn the nonlinear target
almost directly from the expanded features instead of trying to
approximate it linearly from raw tanh states.

**The pattern:** translation helps most when (a) the task requires
nonlinear computation and (b) the reservoir is large enough to support
meaningful quadratic features. At DIM=5 (only 80 translation features),
gains are modest or absent. From DIM=6 upward, the benefit grows
monotonically.

## Computational cost

The translation is **O(N) per timestep** — three passes over the N
states (copy, square, multiply). This is negligible compared to the
reservoir step cost of O(N * DIM) for the weighted sum + tanh.
The transform is OpenMP-parallelized for large sample counts
(threshold: 256 samples).

**Memory cost:** 2.5N floats per collected timestep. For DIM=10
(N=1024), that's 2560 features * 18432 samples * 4 bytes ~ 180 MB
for a full training run. This is the dominant memory cost in the
pipeline.

## Implementation

The translation layer is implemented as two free functions in
`TranslationLayer.h`:

- `TranslationTransform<DIM>(states, num_samples)` — applies the full
  x + x² + x*x' transform, returns a `std::vector<float>` of size
  num_samples * 2.5N.
- `TranslationFeatureCount<DIM>()` — returns 2.5N (constexpr).

Both are header-only templates, instantiated at compile time for the
DIM values used in the project (4-10).
