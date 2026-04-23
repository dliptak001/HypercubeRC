# Reservoir Memory Bottleneck in Character-Level Language Modeling

## The Finding

Empirical sweeps on LM_Text (Tiny Shakespeare, 96-token vocab)
show that **all tested configurations converge to the same performance
ceiling** regardless of reservoir size, readout capacity, or training
budget:

| Configuration | Pass 1 BPC | Final BPC | Top-1 |
|---|---|---|---|
| DIM 13, of=0.5, ch=4, 3 passes | 3.20 | 3.05 | 41.5% |
| DIM 13, of=1.0, ch=8, 10 passes | 3.22 | ~3.05* | ~41%* |
| DIM 12, of=1.0, ch=8, 10 passes | 3.21 | (tracking same) | (tracking same) |
| DIM 10, of=1.0, ch=8, 10 passes | 3.27 | 3.05 | 41.2% |

\*DIM 13 of=1.0/ch=8 run was terminated early; trajectory matched DIM 10.

Every change that should help — 8x more neurons, double the visible
state, double the CNN channels, triple the training passes, higher
learning rate — produces the same final BPC within measurement noise.
The readout has capacity it cannot use.  The bottleneck is upstream:
**the reservoir's representation itself.**

**Conclusion.** The memory depth intrinsic to the hypercube reservoir —
governed by exponential decay at the spectral radius — is too shallow
for language modeling applications. No combination of DIM, readout
capacity, or training budget can compensate: the bottleneck is the
reservoir's fixed-weight dynamics, not the downstream learner. Work on
LM_Text is paused until this is resolved. The next area of
exploration is **reservoir cascades** — chaining multiple reservoirs so
that downstream stages operate on slower, higher-order temporal features,
extending the effective memory horizon without pushing any single
reservoir toward chaotic instability.

## Why the Reservoir Limits Context

In an echo state network, the reservoir's recurrent weights are **fixed
and random**.  Information about past inputs decays exponentially at a
rate governed by the spectral radius:

```
signal_strength(t) ≈ SR^t
```

At SR = 0.90 (the scale-invariant default), a character's influence
drops to:
- 35% after 10 steps
- 12% after 20 steps
- 1.5% after 40 steps
- 0.1% after 65 steps

The reservoir produces a rich nonlinear embedding of **recent** input
history — roughly 20-40 characters — but has no mechanism to selectively
retain information beyond that horizon.  This is an inherent property of
the echo state condition, not a tuning failure.

For character-level language modeling, 20-40 characters of effective
context is enough to learn:
- Character-level statistics (bigrams, trigrams)
- Common short-word completions (`th` → `e`, `wh` → `at`)
- Dialogue formatting (`\n` + uppercase → character name + `:`)

But it is insufficient for:
- Word-level coherence (most English words are 4-8 chars, and predicting
  word *boundaries* requires remembering the current word's length)
- Syntactic structure (subject-verb agreement spans 10-30+ chars)
- Dialogue patterns (who is speaking, what was said previously)
- Verse structure (line length, rhyme — requires 40-80 chars of context)

The BPC ceiling of ~3.05 corresponds roughly to a **bigram model** (the
standard bigram baseline for English is ~3.0 BPC).  This is consistent:
the reservoir provides enough memory for strong bigram statistics and
partial trigram coverage, but not enough for the word-level and
phrase-level patterns that drive BPC below 2.5.

## What Controls Memory Depth

Four parameters in `ReservoirConfig` affect how long the reservoir
retains information about past inputs.  DIM is notably absent — the
DIM 10 vs DIM 13 experiments proved that adding more neurons doing the
same dynamics does not extend memory.

### Spectral Radius (SR) — the dominant factor

The spectral radius governs the largest eigenvalue of the recurrent
weight matrix.  Information decays as ~SR^t per timestep.  This is the
primary "clock speed" of forgetting.

| SR | Steps to 10% signal | Steps to 1% |
|----|---------------------|-------------|
| 0.85 | 14 | 28 |
| 0.90 | 22 | 44 |
| 0.95 | 45 | 90 |
| 0.98 | 114 | 228 |

Higher SR extends memory but pushes toward the edge of chaos.  As shown
in the SR experiment below, SR = 0.98 degraded performance despite the
longer theoretical memory window.  SR = 0.90 is the empirically optimal
balance across all tested tasks.

### Leak Rate — the underexplored knob

The reservoir update equation is:

```
state(t) = (1 - leak) * state(t-1) + leak * tanh(alpha * weighted_sum)
```

At the current default of **leak_rate = 1.0**, the old state term
vanishes entirely — the neuron's activation is fully replaced every
step.  All temporal memory comes from the recurrent weight matrix.

This default dates to Jaeger's original ESN formulation (2001), which
did not include a leaky integrator.  It is the right choice for fast-
dynamics tasks (NARMA-10, signal classification) where the reservoir
must respond immediately to new inputs and the relevant context is
short.  HypercubeRC's benchmarks and scale-invariance results were all
established at leak_rate = 1.0 because those tasks reward fast response.

For language modeling, leak_rate = 1.0 is likely too aggressive.  The
reservoir overwrites its entire state every step, discarding information
that a slower integrator would preserve.  Lowering leak_rate to 0.3-0.5
would blend each new activation with 50-70% of the previous state,
effectively giving the reservoir a "momentum" that extends memory through
a different mechanism than SR:

- **SR** controls how much past state survives the recurrent weight
  multiplication.  Raising SR toward 1.0 risks chaos.
- **Leak rate** controls how much of the old state is *kept verbatim*
  before the recurrent computation even runs.  Lowering it is a
  stability-neutral way to slow the reservoir's time constant.

The two mechanisms are complementary.  SR = 0.90 with leak_rate = 0.3
could provide effective memory comparable to SR = 0.98 at leak_rate = 1.0
without the chaotic instability — the past state is preserved by
blending, not by amplifying recurrent dynamics.

**This combination has not been tested on LM_Text** and is the
most promising single-parameter experiment for breaking the BPC ceiling.

### Input Scaling — indirect memory effect

Input scaling (default 0.02) controls how strongly each new character
perturbs the reservoir state.  Lower values preserve more of the
existing state, indirectly extending memory.  At 0.02 it is already
conservative — the input is a gentle nudge, not a hard reset.
Significant further reduction would make the reservoir unresponsive
to new inputs.

### Alpha — activation gain

Alpha (default 1.0) scales the argument to tanh.  Lower alpha keeps
activations in the linear regime of tanh, making dynamics more linear
and memory longer (a perfectly linear system has infinite memory up to
its rank).  Higher alpha drives saturation, increasing nonlinearity at
the cost of information preservation.  Reducing alpha trades expressive
power for memory — useful if the task needs longer context more than it
needs complex nonlinear features.

## The Spectral Radius Experiment

An obvious hypothesis: push SR higher to extend the memory window.
At SR = 0.98, the theoretical memory horizon extends to ~114 steps
(5x longer than SR = 0.90).

**Result: SR = 0.98 made things worse.**

| SR | Pass 1 Top-1 | Pass 1 BPC |
|----|------|------|
| 0.90 | 37.6% | 3.27 |
| 0.98 | 31.6% | 3.59 |

Accuracy dropped 6 percentage points and BPC increased by 0.32.  Near
the edge of chaos (SR → 1.0), reservoir dynamics become increasingly
sensitive to perturbations.  The state representations are noisier and
harder for the readout to decode.  The longer memory window exists in
theory but is drowned in chaotic dynamics that the fixed-weight reservoir
cannot control.

**This rules out simple SR tuning as a path to better language modeling.**
The memory/stability tradeoff in a fixed-weight reservoir is fundamental:
longer memory requires higher SR, but higher SR degrades state quality.
SR = 0.90 is the empirically optimal balance — consistent with the
scale-invariant finding from NARMA-10 benchmarks.

## How Trained Sequence Models Differ

The reservoir's limitation is not its architecture but the fact that its
recurrent weights are **untrained**.  Comparing with models that achieve
BPC < 2.0 on the same task:

### LSTM (~1.2 BPC)

LSTMs also process one token per step with recurrent state, but their
gating mechanism is **learned** via backpropagation through time.  The
forget gate learns to selectively retain information across hundreds of
steps when it matters for prediction.  The key difference:

- **Reservoir**: information decays at a fixed rate (SR^t) regardless of
  whether it is useful.  The reservoir cannot learn that the letter `q`
  should "remember" to emit `u` — the `q` signal simply fades at the
  same rate as everything else.
- **LSTM**: the forget gate learns task-specific retention.  It can hold
  a `q` signal at full strength until `u` is emitted, then release it.
  This selective memory is why LSTMs achieve word-level and phrase-level
  coherence.

### Transformer (~1.0-1.5 BPC)

Transformers process the entire context window in parallel via
self-attention.  Every token can attend to every other token — there is
no recurrence, no memory decay, and no echo state horizon.  Position 0
is exactly as accessible as position N-1.

The reservoir's streaming one-step-at-a-time processing is fundamentally
different from attention's random-access over the full context.  No
readout architecture, no matter how powerful, can recover context that
the reservoir has already lost to exponential decay.

### The Core Distinction

In both LSTM and Transformer architectures, the model **learns how to
use context** — which information to retain, which to discard, and how
far back to look.  The reservoir provides context through fixed dynamics
that cannot adapt to the task.  The CNN readout is trained, but it sees
only a single snapshot of the current reservoir state — it cannot reach
back in time to attend to inputs the reservoir has already forgotten.

## Implications for HypercubeRC

### What this validates

- The streaming online training pipeline works correctly and trains
  efficiently.
- HypercubeCNN as a readout successfully learns from reservoir state and
  achieves the theoretical ceiling for this representation.
- Scale invariance extends to language modeling: DIM 10 (1024 neurons)
  matches DIM 13 (8192 neurons) at the same BPC ceiling, confirming that
  reservoir dynamics — not size — determine representation quality.
- The readout is not the bottleneck.  Doubling channels, doubling
  visible state, and adding training passes produce no improvement.

### What this means for the LM_Text example

BPC ~3.05 with ~41% top-1 accuracy is the ceiling for a single-character
streaming ESN on this task.  The model learns character-level statistics
and dialogue formatting but cannot produce coherent words.  This is a
legitimate and informative result, not a failure — it precisely
characterizes what fixed-weight reservoir dynamics can and cannot capture
in natural language.

### Directions that could break the ceiling

These would require architectural changes, not hyperparameter tuning:

1. **Multi-character input injection.**  Feed char(t) and char(t-1) (or
   a wider window) as simultaneous inputs.  This gives the readout
   explicit bigram/trigram context without relying on the reservoir's
   fading memory.  Cost: doubles input channels (halves vertices per
   channel at fixed DIM).

2. **Reservoir cascades.**  Chain multiple reservoirs so that
   downstream stages operate on the output of upstream stages,
   extracting slower, higher-order temporal features.  Each reservoir
   remains fixed-weight (no gradient), but the cascade extends the
   effective memory horizon without pushing any single reservoir
   toward chaotic instability.

3. **Hybrid architecture.**  Train the reservoir weights (or a subset)
   via backpropagation through time, converting the ESN into a standard
   RNN with hypercube topology.  This abandons the reservoir computing
   paradigm but could leverage the hypercube's structural advantages
   (XOR addressing, scale-invariant connectivity) in a fully trainable
   network.

### Tasks where the reservoir excels

The reservoir memory bottleneck is specific to tasks requiring long-range
sequential context.  For tasks where the relevant context fits within the
echo state horizon (~20-40 steps at SR = 0.90), the reservoir + HCNN
readout is highly effective:

- **NARMA-10** (10-step nonlinear memory): NRMSE 0.122 at DIM 10
- **Signal classification** (waveform recognition): near-perfect accuracy
- **Streaming anomaly detection**: real-time with minimal latency
- **Time series prediction** with short-horizon dependencies

These tasks have context requirements well within the reservoir's memory
window, and the HCNN readout's ability to discover nonlinear features in
the reservoir state provides genuine advantages over linear readouts.
