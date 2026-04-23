# Signal Classification — Waveform Recognition

## What this example demonstrates

The reservoir acts as a feature extractor for pattern recognition.
Four waveform types — sine, square, triangle, chirp — are fed to the
reservoir in alternating blocks. The HCNN readout classifies which
waveform is active at each timestep, using only the reservoir's
internal state.

This example focuses on multi-class classification, with a confusion
matrix and transition dynamics analysis.

## Conceptual background

Reservoir computing is often presented as a time series prediction tool,
but the reservoir state is also a powerful feature vector for classification.
At any given timestep, the N-dimensional state encodes the recent input
history — and different waveforms produce different trajectories through
state space.

The key insight: you don't need to design features by hand. The reservoir's
nonlinear dynamics and fading memory automatically transform the raw input
into a high-dimensional representation where different signal classes become
separable.

**HCNN: native multi-class.** The CNN readout supports multi-class
natively via `num_outputs=4` and `ReadoutTask::Classification`, using
softmax + cross-entropy loss. A single readout handles all four classes.

## The four waveforms

| Waveform | Frequency | Character |
|----------|-----------|-----------|
| Sine | 0.11 | Smooth, continuous |
| Square | 0.13 | Discontinuous jumps |
| Triangle | 0.12 | Piecewise linear |
| Chirp | 0.10 | Accelerating frequency |

Frequencies are deliberately close together and 15% uniform noise is
added to the signal. This forces the readout to classify by waveform
shape rather than frequency alone.

## The pipeline

```
Waveform blocks ──> Reservoir ──> HCNN: 4-class softmax ──> Class
  40 steps each      32 neurons     (all 32 vertices)        0,1,2,3
                     (fixed)
```

**Step by step:**

1. **Generate signal** — 75 full cycles through all 4 waveforms (300 blocks
   total, 40 steps each = 12,000 timesteps). Each block starts at phase 0.
   Uniform noise in [-0.15, +0.15] is added to every sample.

2. **Warmup** — 300 steps of sine to wash out initial conditions.

3. **Collect** — 12,000 steps with per-step class labels.

4. **Train** — Single 4-class HCNN readout on 70% of the data, trained
   with Adam and cosine LR schedule.

5. **Evaluate** — Confusion matrix, per-class accuracy, and transition
   dynamics (how quickly the reservoir locks onto a new waveform after a
   block switch).

## What to expect

### Default configuration

DIM=5, 32 neurons, leak_rate=0.65, readout on all 32 vertices.
The leak rate is intentionally detuned from the optimal 0.35 to produce
visible classification errors — at 0.35 the readout achieves 100%.
Close frequencies + noise + short blocks + high leak rate make this
a non-trivial classification challenge.

**Overall accuracy: ~98.9%**

| Class | Accuracy | Notes |
|-------|----------|-------|
| Sine | 98.8% | Slight confusion with square (0.9%) |
| Square | 99.4% | Near-perfect |
| Triangle | 99.8% | Near-perfect |
| Chirp | 97.8% | Hardest class — confused with triangle (1.2%) and sine (0.9%) |

Chirp is the hardest class because its accelerating frequency starts
slowly, resembling both sine and triangle in the early part of each
block before the sweep becomes distinctive.

### Transition lock-on

| Steps after switch | Accuracy |
|--------------------|----------|
| 0-3 | 95.9% |
| 0-5 | 97.6% |
| 0-10 | 98.7% |
| 0-20 | 98.7% |
| Entire block | 98.9% |

The reservoir needs ~5 steps to lock onto a new waveform after a block
transition. At leak_rate=0.65, neurons retain 35% of their previous
state, so old waveform dynamics persist briefly into the new block.

### Effect of leak rate on classification

The leak rate controls the tradeoff between classification accuracy and
transition agility.

**Lower leak rates (0.3-0.5):** Neurons retain more history, improving
steady-state accuracy but slowing lock-on after transitions. Better for
long blocks where classification accuracy matters more than speed.

**Higher leak rates (0.7-1.0):** Faster lock-on but less temporal context.
Better for short blocks or fast-changing inputs.

## Things to try

- **Leak rate.** Set `cfg.leak_rate` in the source. The default is 0.65.
  Try 0.3 (slower transitions, potentially higher steady-state accuracy)
  or 1.0 (instant lock-on, less context).

- **Block size.** The default is 40 steps. Try 150 for an easier task
  (more context per block) or 20 for a harder one (transition zone
  dominates each block).

- **Noise level.** `NOISE_LEVEL` is 0.15. Raise to 0.25+ for a harder
  task, or remove noise entirely to see how much it affects accuracy.

- **DIM.** The default is DIM=5 (32 neurons). Try DIM=7 (128 neurons)
  for near-perfect results, showing how reservoir capacity affects
  classification.

- **HCNN epochs.** Default is 25. Try 100 for higher accuracy, or 5
  to see undertrained behavior.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/SignalClassification
```
