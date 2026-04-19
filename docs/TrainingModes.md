# Training Modes: Batch vs Streaming

HypercubeRC supports two training modes for the CNN readout: **batch**
and **streaming**. Both drive the reservoir through a continuous text
corpus and train the CNN head to predict the next character. They
differ in when and how the CNN weights are updated.

## Batch Training

The original mode. Collect all reservoir states into RAM, then train
the CNN readout over multiple epochs.

```
Warmup(64)  →  Run(train+val chars)  →  Train(epochs × batches)
                   ↓                         ↓
             states buffer (RAM)       CNN epoch loop
             N floats × positions      shuffled mini-batches
```

### Pipeline

1. **Warmup**: drive reservoir through 64 chars, no state collection.
2. **Run**: drive reservoir through `train_chars + val_chars` positions.
   Stores the full N-dimensional state at every step into a contiguous
   `states_` buffer.
3. **Subsample**: `HCNNStates()` extracts the `output_fraction` subset
   (e.g., 512 of 4096 vertices) into a temporary buffer for CNN training.
4. **Train**: the CNN readout iterates over stored states in shuffled
   mini-batches for `epochs` passes.  Cosine LR decay across epochs.
5. **Eval**: per-epoch hooks score val states via `PredictRaw()` on
   stored positions, computing top-k accuracy, BPC, and text samples.

### RAM Cost

Dominated by the states buffer:

| DIM | N     | 1M positions  | +CNN copy (of=0.125) | Peak     |
|-----|-------|---------------|---------------------|----------|
| 12  | 4096  | 15.6 GiB      | ~2 GiB              | ~18 GiB  |
| 13  | 8192  | 30.5 GiB      | ~3.5 GiB            | ~34 GiB  |
| 14  | 16384 | 61 GiB        | ~7 GiB              | ~68 GiB  |

The full N-dimensional state is stored even when `output_fraction < 1`,
because the library stores states generically before readout-specific
subsampling.  Half or more of the stored data may never be read.

### Strengths

- **Multiple epochs** over the same data.  The CNN sees each state
  many times, with shuffled ordering for gradient diversity.
- **Batch normalization** in the CNN benefits from full-dataset
  statistics computed per epoch.
- **Input standardization** computed exactly from all training states.
- **Rich eval diagnostics**: stored states allow per-position
  `PredictRaw()` for BPC, top-k, and per-class confusion.

### Weaknesses

- **RAM scales with positions x N**.  DIM 13+ at 900k positions
  exceeds 32 GiB.  DIM 14 requires 64+ GiB.
- **Overfitting**.  Experiments show val metrics plateau by epoch 10
  and degrade afterward — 90% of compute is wasted on memorization.
- **All-or-nothing**.  Cannot evaluate until all states are collected
  and at least one training pass completes.

### Config (batch mode)

```cpp
// Config.h — batch mode (DIM 12 example)
kDIM = 12;
train_chars = 900000;
val_chars   = 100000;
epochs      = 15;         // model peaks at ~10
batch_size  = 4096;
output_fraction = 1.0f;   // or 0.5, 0.25, 0.125
```

### API

```cpp
esn.Run(bits, train_chars + val_chars);    // collect all states
esn.Train(targets, train_chars, cnn_cfg, hooks);  // epoch loop
esn.Accuracy(labels, start, count);        // score stored states
esn.PredictRaw(timestep, logits);          // predict from stored state
```

---

## Streaming Training

Online mode. Train the CNN one gradient step per character as the
reservoir processes the corpus.  No states buffer.

```
Warmup(64)  →  InitOnline(32k states)  →  Stream(train_chars)  →  Stream(val_chars)
                      ↓                        ↓                       ↓
               standardization           TrainLiveStep()        PredictLiveRaw()
               + architecture            per-char CNN update    accumulate metrics
```

### Pipeline

1. **Warmup**: drive reservoir through 64 chars (same as batch).
2. **InitOnline**: `esn.InitOnline()` drives the reservoir through
   `warmup_train_chars` (default 32768) positions, collects the
   states transiently, computes input standardization (per-vertex
   mean and 1/std), builds the CNN architecture, and frees the
   states buffer.  The reservoir is now at position
   `warmup_chars + warmup_train_chars`, ready for streaming.
3. **Stream training**: for each of `train_chars` characters:
   - Encode char as bipolar bits.
   - `esn.Warmup(bits, 1)` — advance reservoir one step.
   - `esn.TrainLiveStep(target_class, lr)` — subsample live state,
     standardize, run one CNN `TrainStep()` (forward + backward +
     Adam update).
   - Discard the state.  Nothing stored.
4. **Stream evaluation**: for each of `val_chars` characters:
   - `esn.Warmup(bits, 1)` — advance reservoir.
   - `esn.PredictLiveRaw(logits)` — predict from live state.
   - Accumulate top-k accuracy, BPC, per-class confusion.
   - No weight updates.
5. **Save model**.

### RAM Cost

Two phases with different RAM profiles:

**Transient (during InitOnline):**

| Component               | Size                       | Example (DIM 13, of=0.5) |
|-------------------------|----------------------------|--------------------------|
| states_ buffer          | warmup_train × N × 4      | 1 GiB                    |
| HCNNStates copy         | warmup_train × of × N × 4 | 512 MiB                  |
| **Transient peak**      |                            | **~1.5 GiB**             |

Freed immediately after standardization is computed.

**Steady-state (during streaming training and eval):**

| Component               | Size                     | Example (DIM 13, of=0.5) |
|-------------------------|--------------------------|--------------------------|
| Reservoir state         | N × 4 bytes              | 32 KiB                   |
| Subsampled scratch      | of × N × 4 bytes         | 16 KiB                   |
| CNN weights + optimizer | architecture-dependent   | ~10 MiB                  |
| **Steady-state total**  |                          | **< 50 MiB**             |

Compared to 34 GiB for batch mode at DIM 13.  DIM 14, 15, 16 are all
feasible with no RAM constraint.

### Strengths

- **Constant RAM** regardless of corpus size or DIM.  Enables DIM 13+.
- **Immediate feedback** — the model learns as it reads.  No waiting
  for a full collection pass before training starts.
- **Natural for sequential data**.  The reservoir processes the corpus
  in order; streaming training updates the CNN in the same order.
- **Simple code path**.  No states buffer, no HCNNStates copy, no
  epoch loop.

### Weaknesses

- **Standardization is approximate**.  Computed from 32k warmup
  states, not the full 900k.  The distribution may shift across the
  corpus.  Increasing `warmup_train_chars` improves accuracy at
  trivial RAM cost.

### Config (streaming mode)

```cpp
// Config.h — streaming mode (DIM 13 example)
kDIM = 13;
warmup_train_chars = 32768; // for standardization
train_chars = 900000;
val_chars   = 100000;
spectral_radius  = 0.90f;
output_fraction  = 0.5f;    // CNN sees 4096 of 8192 vertices
cnn_num_layers   = 1;
cnn_conv_channels = 4;
lr_max = 0.0015f;
```

### API

```cpp
esn.InitOnline(warmup_bits, warmup_count, cnn_cfg);  // build + standardize
esn.Warmup(step_bits, 1);                             // advance reservoir
esn.TrainLiveStep(target_class, lr);                   // online CNN update
esn.PredictLiveRaw(logits);                            // predict from live state
```

---

## Side-by-Side Comparison

| Aspect               | Batch                          | Streaming                      |
|----------------------|--------------------------------|--------------------------------|
| States in RAM        | All positions × N              | 1 state (live)                 |
| RAM at DIM 13        | ~34 GiB                        | < 100 MiB                     |
| Max practical DIM    | 12 (32 GiB) / 13 (64 GiB)     | 16+ (no RAM limit)            |
| Passes over data     | Multiple (epochs, shuffled)    | Multiple (sequential re-feed)  |
| Gradient diversity   | Shuffled mini-batches          | Sequential (corpus order)      |
| Standardization      | Exact (all states)             | Approximate (warmup buffer)    |
| Time to first eval   | Collection + 1 epoch           | After streaming through val    |
| Overfitting risk     | High (model peaks at epoch 10) | Lower (no repeated shuffled batches) |
| Code complexity      | Higher (hooks, checkpoints)    | Lower (linear loop)            |
| Eval flexibility     | Score any stored position      | Must stream through val data   |

## When to Use Each

**Batch** is appropriate when:
- DIM fits in RAM (DIM 12 at 32 GiB, DIM 13 at 64 GiB).
- You want to sweep hyperparameters using mid-training eval hooks.
- You need exact input standardization.
- You want to experiment with multiple epochs and early stopping.

**Streaming** is appropriate when:
- DIM 13+ (RAM constraint makes batch infeasible or wasteful).
- The model plateaus in one pass (our DIM 12 experiments confirm this).
- You want fast iteration — streaming starts training immediately.
- You want simple, predictable resource usage.

## Experiment History

DIM 12 batch experiments (all at 900k train / 100k val):

| Config                  | Val top-1 (epoch 10) | BPC   | Notes                    |
|-------------------------|---------------------|-------|--------------------------|
| nl=1/ch=8/of=0.125     | 37.0%               | 3.25  | Baseline                 |
| nl=2/ch=16/of=0.125    | 40.4%               | 3.09  | Deeper head helps        |
| nl=2/ch=16/of=0.25     | 44.1%               | 2.89  | More spatial signal      |
| nl=1/ch=4/of=1.0       | 44.5%               | 2.89  | Full hypercube, lean head|
| SR=0.95 (any config)   | 32.6%               | 3.49  | Worse — SR=0.90 optimal  |

All batch configs plateau by epoch 10-20.  Val BPC degrades after
epoch 10 due to overfitting (e.g., 2.89 → 3.06 by epoch 90 for the
of=1.0 config).  This motivated the streaming approach: the reservoir
states contain the useful signal, and the CNN extracts it quickly.
Streaming avoids the massive RAM cost of storing states while enabling
larger reservoirs (DIM 13+) that may contain richer signal.
