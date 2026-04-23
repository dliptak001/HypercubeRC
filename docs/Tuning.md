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
`leak_rate`, and HCNN hyperparameters.

---

## Quick-start decision tree

```
1. Pick DIM
   - Prototyping / embedded:  DIM 5-6   (32-64 neurons)
   - Standard tasks:          DIM 7-8   (128-256 neurons)
   - High-capacity research:  DIM 9-16  (512-65536 neurons)

2. Pick HCNN config
   - Start with HRCCNNBaseline<DIM>() from HCNNPresets.h
   - Override epochs for your task: ~25-100 for smooth signals, ~2000 for chaotic
   - Classification: set task=ReadoutTask::Classification, num_outputs=num_classes

3. Set output_fraction (DIM 9+)
   - DIM 9:   0.5   (256 selected vertices)
   - DIM 10:  0.25  (256 selected vertices)
   - DIM 11+: 0.125 or lower
```

---

## Parameter-by-parameter guide

### DIM -- reservoir size

DIM controls the number of neurons (N = 2^DIM). More neurons = more
computational capacity and richer state for the HCNN readout.

| DIM | N     | Best for |
|-----|-------|----------|
| 5   | 32    | Prototyping, embedded, unit tests |
| 6   | 64    | Light benchmarks, fast iteration |
| 7   | 128   | Standard benchmarks, production (simple tasks) |
| 8   | 256   | Production, complex time series |
| 9   | 512   | Research, high-capacity tasks |
| 10  | 1024  | Research (use output_fraction) |
| 11+ | 2048+ | Research (must use output_fraction) |

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
- `SignalClassification`: leak_rate = 0.65 (intentionally detuned to show classification errors; 0.35 gives perfect accuracy)

---

### alpha -- activation gain

Controls the tanh nonlinearity: `tanh(alpha * weighted_sum)`.

| Value | Behavior | Use when |
|-------|----------|----------|
| < 1.0 | Near-linear activation | Task is approximately linear; you want a linear reservoir |
| **1.0** | **Default. Standard tanh** | **Most tasks** |
| > 1.0 | Sharper saturation | You need stronger nonlinear separation (e.g., classification) |
| 2.0+ | Hard limiter (approaches sign function) | Rarely useful; reduces effective dynamic range |

**Most users never touch this.** The HCNN readout provides its own
nonlinear feature discovery via learned convolution kernels.

---

### output_fraction -- readout input control

Controls what fraction of the N reservoir neurons are used as readout
input. Stride-selects a subset of vertices.

| Value  | Selected vertices M | Stride |
|--------|---------------------|--------|
| 1.0    | N (all)             | 1      |
| 0.5    | N/2                 | 2      |
| 0.25   | N/4                 | 4      |
| 0.125  | N/8                 | 8      |
| 0.0625 | N/16                | 16     |

**Valid values:** output_fraction must yield a power-of-2 stride.
Use values from `{1.0, 0.5, 0.25, 0.125, 0.0625, ...}`. Other values
(e.g., 0.1, 0.3) will throw at construction time.

**When to reduce it:**
- DIM >= 9: 0.25-0.5 keeps HCNN training practical
- DIM >= 10: 0.125-0.25 is typical
- DIM >= 11: 0.0625-0.125 to keep training times reasonable

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

2. **Use surveyed seeds for benchmarks.** `SurveyedSeed<DIM>()` in
   `Reservoir.h` returns the per-DIM 500-seed survey winner. Always
   use these for benchmarks and comparisons.

3. **Screen seeds for production.** Run your task across 50-500 seeds
   and pick a top performer. Seed rankings are stable across nearby SR
   values (Spearman correlation > 0.82 in the 0.85-0.90 corridor).

**How much does seed matter?**

At DIM 7, SR=0.90, the 500-seed survey shows:
- MG NRMSE: median 0.0049, best 0.0032, worst 0.0096 (3.0x spread)
- The top tier of seeds clusters in a tight band near the best

Seed selection is free performance -- same compute cost, better results.

---

## HCNN readout tuning

The HCNN readout is the only trained component. See
[docs/Readout.md](Readout.md) for architecture details.

### Starting config

Use `HRCCNNBaseline<DIM>()` from `HCNNPresets.h` as the starting
point. It provides surveyed reservoir seed + baseline CNN config per DIM.

### Key HCNN parameters

| Parameter | Baseline | Tuning guidance |
|-----------|----------|-----------------|
| `epochs` | 2000 | ~25-100 for smooth/periodic signals; ~2000 for chaotic (MG, NARMA) |
| `lr_max` | 0.0015 | Keep <= 0.005; 0.0015 transfers well across DIM |
| `batch_size` | 1<<(DIM-1) | Target ~50k total gradient updates |
| `conv_channels` | 8 | 8 is the baseline; 16-24 for task-specific tuning |
| `num_layers` | 1 | Auto-rule: min(DIM-2, 2); override for task-specific tuning |
| `weight_decay` | 0.0 | Start at 0; only add if overfitting |
| `seed` | per-DIM | CNN weight-init seed; per-DIM winners in HCNNPresets.h |

### Epoch selection

The biggest tuning decision. Smooth signals saturate almost instantly;
chaotic signals need thousands of epochs:

- **Smooth/periodic** (sine, classification, anomaly detection): 25-100 epochs
- **Chaotic/nonlinear** (NARMA-10, Mackey-Glass): 1000-2000 epochs

Don't use the default 200 for either case — it's wrong for both.

---

## Training data requirements

| Metric | Minimum | Recommended | Why |
|--------|---------|-------------|-----|
| Warmup steps | 50 | 200-500 | Wash out initial transient (zero state) |
| Training samples | 5 * N | 18 * N | More data = better generalization |
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
   samples as reservoir neurons?
3. **Increase DIM.** More neurons = more capacity. Go from 7 to 8.
4. **Screen seeds.** Try 50 seeds and pick the best.
5. **Increase HCNN epochs.** If training on a chaotic signal, ensure
   epochs >= 1000.
6. **Try more conv channels.** Increase from 8 to 16 or 24.

### "Training is too slow"

1. **Reduce output_fraction.** Going from 1.0 to 0.5 significantly
   reduces HCNN input dimensionality.
2. **Reduce DIM.** If DIM 8 is too slow, DIM 7 may be sufficient.
3. **Lower epochs.** For smooth signals, 25-100 epochs is usually enough.
4. **Increase batch_size.** Fewer gradient updates per epoch.

### "Predictions are noisy / jittery"

1. **Lower leak_rate.** Try 0.3-0.5 to smooth the reservoir dynamics.
2. **Increase warmup.** Transient artifacts cause jitter in early states.
3. **Increase training size.** More data stabilizes the readout.

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

- **alpha** unless you've exhausted other options. Alpha adjustments
  have second-order effects; addressing DIM or HCNN config first is
  almost always more productive.

- **Reservoir internals.** The connectivity pattern (shell masks +
  nearest neighbors) is not configurable and doesn't need to be.
  The hypercube topology is the architecture's core contribution.

---

## Tuning workflow summary

```
1. Start with defaults:
   ReservoirConfig cfg;
   cfg.seed = SurveyedSeed<8>();
   ESN<8> esn(cfg);

2. Get a baseline:
   auto cnn_cfg = hcnn_presets::HRCCNNBaseline<8>().cnn;
   esn.Warmup(data, 500);
   esn.Run(data + 500, total);
   esn.Train(targets, train_size, cnn_cfg);
   R2 = esn.R2(targets, train_size, test_size);

3. If R2 is insufficient:
   - Increase DIM (7 -> 8 -> 9)
   - Screen 50+ seeds
   - Increase epochs (for chaotic signals)
   - Lower leak_rate if signal is slow
   - Try more conv channels (8 -> 16 -> 24)

4. If speed is insufficient:
   - Reduce output_fraction
   - Decrease DIM
   - Lower epochs (for smooth signals)

5. For production:
   - Screen 100-500 seeds at your target DIM
   - Validate on held-out data
```
