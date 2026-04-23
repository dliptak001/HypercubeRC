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
// Illustrative batch config (DIM 12 example).
// Current Config.h uses streaming mode exclusively.
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

Online mode. The reservoir advances one character at a time;
subsampled states are accumulated into mini-batches and flushed as
parallelized gradient updates.  No full states buffer needed.

```
Warmup(64)  →  InitOnline(32k)  →  [Stream(train) → Eval(val)] × num_passes  →  Save
                     ↓                     ↓                ↓
              standardization      CopyLiveState() +  PredictLiveRaw()
              + architecture       TrainLiveBatch()   accumulate metrics
```

### Pipeline

1. **Warmup**: drive reservoir through `warmup_chars` (64) characters
   without recording state.
2. **InitOnline**: `esn.InitOnline()` drives the reservoir through
   `warmup_train_chars` (default 32768) positions, collects the
   states transiently, computes input standardization (per-vertex
   mean and 1/std), builds the CNN architecture, and frees the
   states buffer.  The reservoir is now at position
   `warmup_chars + warmup_train_chars`, ready for streaming.
3. **Stream training** (repeated for `num_passes`): for each of
   `train_chars` characters:
   - Encode char as bipolar bits.
   - `esn.Warmup(bits, 1)` — advance reservoir one step.
   - `esn.CopyLiveState(buf)` — snapshot the subsampled state into
     an accumulation buffer.  Record the target class.
   - When `mini_batch_size` samples have accumulated, flush via
     `esn.TrainLiveBatch(states, targets, count, lr)` — standardize,
     then one parallelized Adam update across all samples in the batch.
   - Learning rate decays linearly from `lr_max` to
     `lr_max * lr_floor_frac` across all batches in all passes
     (global schedule, no per-pass reset).
4. **Mid-pass evaluation** (after each pass): save the reservoir's
   live state via `esn.SaveReservoirState()`, stream through
   `val_chars` positions scoring top-k accuracy and BPC via
   `esn.PredictLiveRaw()`, then restore the reservoir state to resume
   training from where the pass ended.
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

| Component               | Size                       | Example (DIM 13, of=0.5) |
|-------------------------|----------------------------|--------------------------|
| Reservoir state         | N × 4 bytes                | 32 KiB                   |
| Mini-batch accumulator  | mini_batch × of × N × 4   | 8 MiB                    |
| CNN weights + optimizer | architecture-dependent     | ~10 MiB                  |
| **Steady-state total**  |                            | **< 50 MiB**             |

Compared to 34 GiB for batch mode at DIM 13.  DIM 14, 15, 16 are all
feasible with no RAM constraint.

### Strengths

- **Constant RAM** regardless of corpus size or DIM.  Enables DIM 13+.
- **Immediate feedback** — the model learns as it reads.  No waiting
  for a full collection pass before training starts.
- **Natural for sequential data**.  The reservoir processes the corpus
  in order; streaming training updates the CNN in the same order.
- **Multi-pass without RAM cost**.  The `num_passes` parameter
  re-feeds the training region sequentially (reservoir state
  continues, no reset), getting multiple exposures without storing
  states.

### Weaknesses

- **Standardization is approximate**.  Computed from 32k warmup
  states, not the full 900k.  The distribution may shift across the
  corpus.  Increasing `warmup_train_chars` improves accuracy at
  trivial RAM cost.
- **Sequential gradient order**.  Unlike batch mode's shuffled
  mini-batches, streaming presents samples in corpus order.
  Mini-batch accumulation (`mini_batch_size` > 1) amortizes
  per-sample variance but doesn't shuffle.

### Config (streaming mode)

```cpp
// Config.h — streaming mode (DIM 13 example)
kDIM = 13;
warmup_chars       = 64;
warmup_train_chars = 32768;   // states for CNN standardization
train_chars        = 900000;
num_passes         = 3;       // corpus passes (reservoir continues, no reset)
val_chars          = 100000;
spectral_radius    = 0.90f;
leak_rate          = 1.0f;
output_fraction    = 0.5f;    // CNN sees 4096 of 8192 vertices
cnn_num_layers     = 1;
cnn_conv_channels  = 4;
mini_batch_size    = 512;     // gradient accumulation window
lr_max             = 0.0015f;
lr_floor_frac      = 0.5f;   // linear LR decay floor
```

### API

```cpp
esn.InitOnline(warmup_bits, warmup_count, cnn_cfg);   // build + standardize

// Training loop (per character):
esn.Warmup(step_bits, 1);                              // advance reservoir
esn.CopyLiveState(buf + i * state_dim);                // snapshot for batch
// ... accumulate mini_batch_size samples, then:
esn.TrainLiveBatch(states, targets, count, lr);        // parallelized update

// Eval loop (per character):
esn.PredictLiveRaw(logits);                            // predict from live state

// Mid-pass eval with state preservation:
esn.SaveReservoirState(state, output);                 // save before eval
// ... stream val region ...
esn.RestoreReservoirState(state, output);              // restore after eval
```

---

## Side-by-Side Comparison

| Aspect               | Batch                          | Streaming                      |
|----------------------|--------------------------------|--------------------------------|
| States in RAM        | All positions × N              | 1 mini-batch (live)            |
| RAM at DIM 13        | ~34 GiB                        | < 100 MiB                     |
| Max practical DIM    | 12 (32 GiB) / 13 (64 GiB)     | 16+ (no RAM limit)            |
| Passes over data     | Multiple (epochs, shuffled)    | Multiple (sequential re-feed)  |
| Gradient diversity   | Shuffled mini-batches          | Sequential accumulation        |
| LR schedule          | Cosine decay (per epoch)       | Linear decay (global)          |
| Standardization      | Exact (all states)             | Approximate (warmup buffer)    |
| Time to first eval   | Collection + 1 epoch           | After first pass through val   |
| Overfitting risk     | High (model peaks at epoch 10) | Lower (sequential, no shuffle) |
| Code complexity      | Higher (hooks, checkpoints)    | Moderate (accumulation + state save/restore) |
| Eval flexibility     | Score any stored position      | Must stream through val data   |

## When to Use Each

**Batch** is appropriate when:
- DIM fits in RAM (DIM 12 at 32 GiB, DIM 13 at 64 GiB).
- You want to sweep hyperparameters using mid-training eval hooks.
- You need exact input standardization.
- You want to experiment with multiple epochs and early stopping.

**Streaming** is appropriate when:
- DIM 13+ (RAM constraint makes batch infeasible or wasteful).
- You want fast iteration — streaming starts training immediately.
- You want simple, predictable resource usage.
- Multi-pass re-feeding suffices (no shuffled cross-epoch gradient diversity).

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
