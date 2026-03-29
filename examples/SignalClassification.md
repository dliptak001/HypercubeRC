# Signal Classification — Waveform Recognition

## What this example demonstrates

The reservoir acts as a feature extractor for pattern recognition.
Four waveform types — sine, square, triangle, chirp — are fed to the
reservoir in alternating blocks. One-vs-rest linear readouts classify
each timestep by waveform type using only the reservoir state, never
the raw input. This is the only example that performs multi-class
classification, with a confusion matrix and transition dynamics analysis.

## Conceptual background

Reservoir computing is often presented as a time series prediction tool,
but the reservoir state is also a powerful feature vector for classification.
At any given timestep, the N-dimensional state encodes the recent input
history — and different waveforms produce different trajectories through
state space.

The key insight: you don't need to design features by hand. The reservoir's
nonlinear dynamics and fading memory automatically transform the raw input
into a high-dimensional representation where different signal classes become
linearly separable.

**One-vs-rest classification.** Since the readout is linear, multi-class
classification uses the standard one-vs-rest decomposition: train one
readout per class (target = +1 for that class, -1 for all others), then
classify by taking the argmax over the four readout scores.

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
Waveform blocks ──> Reservoir ──> State features ──> 4 Readouts ──> argmax ──> Class
  150 steps each     128 neurons   320 features       one-vs-rest             0,1,2,3
```

**Step by step:**

1. **Generate signal** — 20 full cycles through all 4 waveforms (80 blocks
   total, 150 steps each = 12,000 timesteps). Each block starts at phase 0
   so the reservoir sees the full characteristic shape from the start.

2. **Warmup** — 300 steps of sine to wash out initial conditions.

3. **Collect** — 12,000 steps with per-step class labels.

4. **Extract features** — Translation layer (2.5N = 320 features) is the
   default and strongly recommended for classification. Raw features work
   but accuracy drops significantly.

5. **Train** — Four one-vs-rest Ridge readouts on 70% of the data.

6. **Evaluate** — Confusion matrix, per-class accuracy, and transition
   dynamics (how quickly the reservoir locks onto a new waveform after a
   block switch).

## What to expect

With DIM=7, translation features, Ridge readout:

| Class | Accuracy | Notes |
|-------|----------|-------|
| Sine | 100% | Perfectly separable |
| Square | 100% | Perfectly separable |
| Triangle | ~99% | Near-perfect |
| Chirp | ~88% | Confused with Sine at low sweep frequencies |

**Overall accuracy: ~97%**

The chirp-sine confusion makes physical sense: a chirp starts at a low
frequency similar to the sine's 0.08, so the reservoir states overlap
during the slow-sweep portion of each chirp block.

## Transition dynamics

The most interesting part of the output. When the waveform switches, the
reservoir state still reflects the previous signal. The analysis shows how
many steps it takes to "lock on" to the new waveform:

| Steps after switch | Accuracy |
|--------------------|----------|
| 0-3 | ~75% (3 of 4 carry over) |
| 0-5 | ~75% |
| 0-10 | ~75% |
| 0-20 | ~76% |
| Entire block | ~97% |

The reservoir needs roughly 20 steps to wash out the old dynamics.
After that, steady-state accuracy approaches 100%.

## Things to try

- **Raw vs. translation.** Pass `raw` as a command-line argument.
  Classification accuracy drops sharply — the nonlinear features from
  the translation layer are critical for separating waveform classes.

- **Change block size.** Shorter blocks (e.g., 50 steps) make classification
  harder because a larger fraction of each block is spent in the transition
  zone where the reservoir hasn't locked on yet.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/SignalClassification              # default: translation features
./build/SignalClassification translation  # explicit translation
./build/SignalClassification raw          # raw N features (lower accuracy)
```
