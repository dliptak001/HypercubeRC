# Tuning HypercubeRC

Practical guidance for getting the best performance out of HypercubeRC.
Start with the defaults, then adjust one parameter at a time.

## The defaults work

HypercubeRC's defaults are scale-invariant -- the same configuration
is optimal across all DIM values (see [ScaleInvariance.md](ScaleInvariance.md)):

| Parameter | Default | Why it works everywhere |
|-----------|---------|------------------------|
| `spectral_radius` | 0.90 | Vertex-transitive topology: stability threshold is a local property, independent of N |
| `input_scaling` | 0.02 | Uniform input distribution via stride-interleaving; optimal drive strength doesn't shift with N |

**Start here.** Only tune if your task has specific requirements that the
defaults don't serve. Most users will only need to adjust `DIM`,
`output_fraction`, and possibly `leak_rate`.

---

## Quick-start decision tree

```
1. Pick DIM
   - Prototyping / embedded:  DIM 5-6   (32-64 neurons)
   - Standard tasks:          DIM 7-8   (128-256 neurons)
   - High-capacity research:  DIM 9-12  (512-4096 neurons)

2. Pick readout
   - DIM <= 8:   ReadoutType::Ridge   (closed-form optimum, fast and accurate)
   - DIM 7+:     ReadoutType::HCNN    (when accuracy ceiling matters or classification)
   - Streaming:  ReadoutType::HCNN    (supports online training for incremental adaptation)

3. Pick feature mode
   - Most tasks:         FeatureMode::Translated  (20-70% lower NRMSE)
   - Speed-constrained:  FeatureMode::Raw         (fewer features, faster)

4. Set output_fraction (DIM 9+)
   - DIM 9:   0.5   (256 selected vertices)
   - DIM 10:  0.25  (256 selected vertices)
   - DIM 11+: 0.125 or lower
   Ridge cost is quadratic in feature count -- this is the main scaling lever.
```

---

## Parameter-by-parameter guide

### DIM -- reservoir size

DIM controls the number of neurons (N = 2^DIM). More neurons = more
computational capacity, but also more features for the readout to handle.

| DIM | N    | Features (Translated) | Ridge solve time | Best for |
|-----|------|-----------------------|------------------|----------|
| 5   | 32   | 80                    | instant          | Prototyping, embedded, unit tests |
| 6   | 64   | 160                   | instant          | Light benchmarks, fast iteration |
| 7   | 128  | 320                   | ~1 ms            | Standard benchmarks, production (simple tasks) |
| 8   | 256  | 640                   | ~5 ms            | Production, complex time series |
| 9   | 512  | 1280 (at full)        | ~40 ms           | Research, high-capacity tasks |
| 10  | 1024 | 2560 (at full)        | ~300 ms          | Research (use output_fraction) |
| 11+ | 2048+| 5120+ (at full)       | seconds          | Research (must use output_fraction) |

**Rule of thumb:** increase DIM until test performance plateaus, then stop.
DIM 8 is the sweet spot for most real-world tasks.

---

### spectral_radius -- memory vs. responsiveness

Controls how quickly past inputs fade from the reservoir's memory.

| Value | Behavior | Use when |
|-------|----------|----------|
| 0.80-0.85 | Fast decay, short memory | Input has high frequency content; you only need recent context |
| **0.90** | **Default. Best general-purpose compromise** | **Start here for any task** |
| 0.95 | Longer memory, higher variance | Memory capacity is critical (long-lag recall) |
| 1.00 | Edge of chaos | Not recommended -- variance doubles, seed sensitivity spikes, NARMA degrades |

**The landscape is flat around 0.90.** At DIM 8, sweeping SR from 0.89
to 0.93 changes NARMA NRMSE by less than 0.002. Don't spend time
fine-tuning SR unless you have a specific reason.

**Warning:** SR >= 1.0 is a qualitatively different regime. Rank
correlation with 0.90 drops below 0.45, meaning seeds screened at 0.90
do not transfer. Avoid unless you know what you're doing.

---

### input_scaling -- drive strength

Controls the magnitude of input weights, drawn from
U(-input_scaling, +input_scaling). Determines how strongly external
input perturbs the reservoir state.

| Value | Behavior | Use when |
|-------|----------|----------|
| 0.01 | Very gentle drive | Input signal is large-amplitude or noisy |
| **0.02** | **Default. Optimal across all DIMs** | **Start here** |
| 0.05-0.10 | Stronger drive | Input signal is very weak or you need faster response |
| 0.20+ | Aggressive drive | Rarely beneficial; can saturate tanh and destroy dynamics |

**Interaction with input amplitude:** Inputs are clamped to [-1, +1]
internally. If your raw signal has large amplitude, normalize it before
feeding to the ESN. The optimal input_scaling assumes inputs in a
reasonable range.

---

### leak_rate -- temporal smoothing

Controls the leaky integrator:
```
state = (1 - leak_rate) * old_state + leak_rate * activation
```

| Value | Behavior | Use when |
|-------|----------|----------|
| **1.0** | **Default. Full replacement -- no smoothing** | **Fast-changing signals, standard benchmarks** |
| 0.3-0.5 | Moderate smoothing | Signal has slow dynamics mixed with noise |
| 0.1-0.2 | Heavy smoothing | Very slow dynamics, need long effective memory |

**What it does:** Low leak rates make each neuron a low-pass filter,
blending new activations with the previous state. This effectively
extends the reservoir's temporal horizon without increasing SR.

**When to lower it:**
- Your signal changes slowly relative to the timestep
- You see high test-time variance (prediction jitters)
- You need memory beyond what SR=0.90 provides but SR=0.95 is too unstable

**Interaction with SR:** Leak rate and spectral radius both affect
memory length. Low leak_rate + high SR compounds the memory effect.
If you lower leak_rate, you may also want to lower SR slightly to
avoid instability.

**Examples in the codebase:**
- `BasicPrediction`: leak_rate = 0.2 (slow sine wave, heavy smoothing)
- `StreamingAnomaly`: leak_rate = 0.3 (process monitoring, moderate smoothing)
- `SignalClassification`: leak_rate = 0.35 (waveform classification, moderate smoothing)

---

### alpha -- activation gain

Controls the tanh nonlinearity: `tanh(alpha * weighted_sum)`.

| Value | Behavior | Use when |
|-------|----------|----------|
| < 1.0 | Near-linear activation | Task is approximately linear; you want a linear reservoir |
| **1.0** | **Default. Standard tanh** | **Most tasks** |
| > 1.0 | Sharper saturation | You need stronger nonlinear separation (e.g., classification) |
| 2.0+ | Hard limiter (approaches sign function) | Rarely useful; reduces effective dynamic range |

**Most users never touch this.** The translation layer already provides
nonlinear features (x^2, x*x_antipodal) that compensate for tanh's
information compression. Adjusting alpha is a second-order effect.

---

### output_fraction -- readout cost control

Controls what fraction of the N reservoir neurons are used as readout
features. This is the primary lever for managing Ridge readout cost at
large DIM values.

| Value | Selected vertices M | Features (Translated) | Ridge cost relative |
|-------|--------------------|-----------------------|--------------------|
| 1.0   | N                  | 2.5N                  | 1x (baseline) |
| 0.5   | N/2                | 1.25N                 | ~0.25x |
| 0.25  | N/4                | 0.625N                | ~0.0625x |
| 0.1   | N/10               | 0.25N                 | ~0.01x |

**Ridge cost is quadratic in feature count** (building the Gram matrix
is O(features^2 * samples)). Halving the feature count gives a 4x
speedup.

**When to reduce it:**
- DIM >= 9 (512+ neurons): 0.25-0.5 keeps Ridge practical
- DIM >= 10 (1024+ neurons): 0.1-0.25 is typical
- DIM >= 11 (2048+ neurons): 0.05-0.125 to keep solve times reasonable

**Performance impact:** The hypercube's vertex-transitive topology means
stride-selected subsets are representative. Reducing output_fraction from
1.0 to 0.5 typically costs less than 5% accuracy on standard benchmarks.

---

### seed -- weight initialization

Every seed (including 0) produces a valid reservoir. Different seeds
produce measurably different performance -- this is inherent to random
weight initialization, not a bug.

**Seed selection strategy:**

1. **Don't bother for prototyping.** Any seed works. Use seed=42 or 0.

2. **Screen seeds for production.** Run your task across 50-500 seeds
   and pick a top performer. Seed rankings are stable across nearby SR
   values (Spearman correlation > 0.82 in the 0.85-0.90 corridor), so
   seeds screened at the defaults transfer reliably.

**How much does seed matter?**

At DIM 7, SR=0.90, the 500-seed survey shows:
- MG NRMSE: median 0.0049, best 0.0032, worst 0.0096 (3.0x spread)
- The top tier of seeds clusters in a tight band near the best

Seed selection is free performance -- same compute cost, better results.

---

## Readout tuning

### Ridge lambda

Controls the regularization strength in Ridge regression.

| lambda | Behavior | Use when |
|--------|----------|----------|
| 0.01-0.1 | Weak regularization | Large training set, few features, low noise |
| **1.0** | **Default** | **General-purpose starting point** |
| 10-100 | Strong regularization | Small training set, many features, noisy data |

**Finding the right lambda:**

Lambda trades off bias and variance. Too low: overfitting (train R^2
near 1.0 but test R^2 much lower). Too high: underfitting (both train
and test R^2 are mediocre).

```cpp
// Quick lambda sweep
for (double lambda : {0.01, 0.1, 1.0, 10.0, 100.0}) {
    esn.Train(targets, train_size, lambda);
    double r2 = esn.R2(targets, train_size, test_size);
    printf("lambda=%.2f  R2=%.6f\n", lambda, r2);
}
```

### Linear SGD parameters

| Parameter | Default | Tuning guidance |
|-----------|---------|-----------------|
| `lr` | auto (1/nf) | Increase if training is slow to converge; decrease if loss oscillates |
| `epochs` | 200 | Pocket selection means more epochs rarely hurts, but also rarely helps past 200 |
| `weight_decay` | 1e-4 | Increase for more regularization; decrease if underfitting |
| `lr_decay` | 0.01 | Controls learning rate schedule: lr_effective = lr / (1 + lr_decay * epoch) |

**In practice:** The auto learning rate and 200 epochs work well. If
you need better performance at DIM 7+, switch to Ridge instead of
tuning SGD.

---

## Training data requirements

| Metric | Minimum | Recommended | Why |
|--------|---------|-------------|-----|
| Warmup steps | 50 | 200-500 | Wash out initial transient (zero state) |
| Training samples | 5 * NumFeatures | 18 * N | Ridge needs overdetermined system; more data = better conditioning |
| Test samples | 20% of total | 30% of total | Enough to estimate generalization reliably |

**Warmup is critical.** The reservoir starts from all-zeros. Without
warmup, the first few hundred collected states contain transient
artifacts that degrade training. More warmup never hurts (it doesn't
allocate memory).

---

## Common tuning scenarios

### "My R^2 is low"

1. **Check warmup.** Are you using at least 200 warmup steps?
2. **Check training size.** Do you have at least 10x as many training
   samples as features?
3. **Try Translated mode.** If using Raw, switch to Translated -- it
   typically improves NRMSE by 20-70%.
4. **Increase DIM.** More neurons = more capacity. Go from 7 to 8.
5. **Screen seeds.** Try 50 seeds and pick the best.
6. **Sweep lambda** (Ridge) or check that lr isn't too high (Linear).

### "Training is too slow"

1. **Reduce output_fraction.** This is the most effective lever.
   Going from 1.0 to 0.5 gives ~4x speedup with minimal accuracy loss.
2. **Switch to Raw features.** 2.5x fewer features than Translated.
3. **Reduce DIM.** If DIM 8 is too slow, DIM 7 may be sufficient.
4. **Use Ridge readout.** The closed-form solve is a single pass —
   no epoch tuning needed.

### "Predictions are noisy / jittery"

1. **Lower leak_rate.** Try 0.3-0.5 to smooth the reservoir dynamics.
2. **Increase warmup.** Transient artifacts cause jitter in early states.
3. **Increase training size.** More data stabilizes the readout.
4. **Increase Ridge lambda.** Stronger regularization smooths predictions.

### "I need longer memory"

1. **Lower leak_rate.** This extends effective memory without increasing
   instability. Try 0.2-0.3.
2. **Increase SR to 0.95.** More memory, but higher variance across seeds.
   Screen seeds more carefully.
3. **Increase DIM.** Larger reservoirs have more memory capacity
   (MC scales with N).
4. **Do not go to SR=1.00** unless you're specifically studying
   edge-of-chaos dynamics.

### "I'm building a streaming / anomaly detection system"

1. **Use HCNN online training** for incremental adaptation.
2. **Set leak_rate to 0.2-0.4** for temporal smoothing.
3. **Prime with a large batch** (500+ warmup, 18*N+ training samples).
4. **Use ClearStates() between windows** -- it preserves the readout
   and reservoir state while freeing collected data.

See [StreamingAnomaly example](../examples/StreamingAnomaly.md) for a
complete implementation.

---

## What NOT to tune

- **spectral_radius and input_scaling together.** The defaults are
  jointly optimal. If you change one, the other's optimal value shifts
  in non-obvious ways. Stick with (0.90, 0.02) unless you have a
  specific reason and a sweep tool.

- **alpha** unless you've exhausted other options. The translation layer
  already provides the nonlinear features that alpha adjustments are
  trying to create.

- **Reservoir internals.** The connectivity pattern (shell masks +
  nearest neighbors) is not configurable and doesn't need to be.
  The hypercube topology is the architecture's core contribution.

---

## Tuning workflow summary

```
1. Start with defaults:
   ReservoirConfig cfg;
   cfg.seed = 42;
   ESN<8> esn(cfg);  // Ridge + Translated

2. Get a baseline:
   esn.Warmup(data, 500);
   esn.Run(data + 500, total);
   esn.Train(targets, train_size);
   R2 = esn.R2(targets, train_size, test_size);

3. If R2 is insufficient:
   - Increase DIM (7 -> 8 -> 9)
   - Screen 50+ seeds
   - Sweep Ridge lambda
   - Lower leak_rate if signal is slow

4. If speed is insufficient:
   - Reduce output_fraction
   - Use Raw features
   - Decrease DIM

5. For production:
   - Screen 100-500 seeds at your target DIM
   - Confirm with a lambda sweep
   - Validate on held-out data
```
