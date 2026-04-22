# Signal Classification — Waveform Recognition

## What this example demonstrates

The reservoir acts as a feature extractor for pattern recognition.
Four waveform types — sine, square, triangle, chirp — are fed to the
reservoir in alternating blocks. The HCNN readout classifies which
waveform is active at each timestep, using only the reservoir's
internal state.

This is the only example that performs multi-class classification, with
a confusion matrix and transition dynamics analysis.

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
natively via `num_outputs=4` and `HCNNTask::Classification`, using
softmax + cross-entropy loss. A single readout handles all four classes.

## The four waveforms

| Waveform | Frequency | Character |
|----------|-----------|-----------|
| Sine | 0.08 (slow) | Smooth, continuous |
| Square | 0.25 (fast) | Discontinuous jumps |
| Triangle | 0.15 (medium) | Piecewise linear |
| Chirp | 0.10 (sweep) | Accelerating frequency |

Each has a distinct frequency and dynamic signature. The reservoir's
recent-input trajectory is completely different for each, making the
classes separable from reservoir state alone.

## The pipeline

```
Waveform blocks ──> Reservoir ──> HCNN: 4-class softmax ──> Class
  150 steps each     128 neurons    (all 128 vertices)       0,1,2,3
                     (fixed)
```

**Step by step:**

1. **Generate signal** — 20 full cycles through all 4 waveforms (80 blocks
   total, 150 steps each = 12,000 timesteps). Each block starts at phase 0
   so the reservoir sees the full characteristic shape from the start.

2. **Warmup** — 300 steps of sine to wash out initial conditions.

3. **Collect** — 12,000 steps with per-step class labels.

4. **Train** — Single 4-class HCNN readout on 70% of the data, trained
   with Adam and cosine LR schedule.

5. **Evaluate** — Confusion matrix, per-class accuracy, and transition
   dynamics (how quickly the reservoir locks onto a new waveform after a
   block switch).

## What to expect

### Default configuration (leak_rate = 0.35)

DIM=7, 128 neurons, leak_rate=0.35, all vertices used by HCNN.

| Class | Accuracy | Notes |
|-------|----------|-------|
| Sine | 99.3% | Near-perfect |
| Square | 100% | Perfectly separable |
| Triangle | 98.7% | Near-perfect |
| Chirp | 92.7% | Confused with Square 3% and Triangle 4% |

**Overall accuracy: 97.7%**

Transition lock-on:

| Steps after switch | Accuracy |
|--------------------|----------|
| 0-3 | 50.0% |
| 0-5 | 60.0% |
| 0-10 | 77.5% |
| 0-20 | 88.8% |
| Entire block | 97.7% |

### Effect of leak rate on classification

The leak rate controls the tradeoff between classification accuracy and
transition agility.

**Better chirp separation at lower leak rates.** Chirp is the hardest
class because it starts at a low frequency similar to sine's 0.08. With
a lower leak rate, the neurons retain recent frequency history, so the
readout can distinguish chirp's accelerating sweep from sine's constant
frequency.

**Slower lock-on after transitions.** When the waveform switches, the
leaky neurons retain state from the previous signal. Lower leak rates
mean old state persists longer — lock-on starts at 50% and climbs through
77.5% by step 10 as the new signal gradually overwrites the old.

**The tradeoff is task-dependent.** For applications where classification
accuracy matters more than transition speed (long blocks, stable signals),
lower leak rates win. For applications where rapid switching is critical
(short blocks, fast-changing inputs), higher leak rates are better.

## Things to try

- **Leak rate.** Set `cfg.leak_rate` in the source. The default for this
  example is 0.35. Try 1.0 (instant lock-on, lower accuracy) or 0.2
  (higher accuracy, slower transitions).

- **Change block size.** Shorter blocks (e.g., 50 steps) make classification
  harder because a larger fraction of each block is spent in the transition
  zone where the reservoir hasn't locked on yet.

- **HCNN epochs.** Default is 100 — this task saturates fast. Try 25 to
  verify saturation, or 200 to confirm no further gain.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/SignalClassification
```
