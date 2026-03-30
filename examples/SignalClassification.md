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

### Leak rate = 1.0 (full replacement)

DIM=7, translation features, Ridge readout, leak_rate=1.0:

| Class | Accuracy | Notes |
|-------|----------|-------|
| Sine | 100% | Perfectly separable |
| Square | 100% | Perfectly separable |
| Triangle | 99.3% | Near-perfect |
| Chirp | 88.0% | Confused with Sine 9% of the time |

**Overall accuracy: 96.8%**

Transition lock-on:

| Steps after switch | Accuracy |
|--------------------|----------|
| 0-3 | 75.0% |
| 0-5 | 75.0% |
| 0-10 | 75.0% |
| 0-20 | 76.2% |
| Entire block | 96.8% |

### Leak rate = 0.6 (leaky integrator)

Same configuration with leak_rate=0.6:

| Class | Accuracy | Notes |
|-------|----------|-------|
| Sine | 99.3% | Near-perfect |
| Square | 100% | Perfectly separable |
| Triangle | 98.7% | Near-perfect |
| Chirp | 96.7% | Substantially improved |

**Overall accuracy: 98.7%**

Transition lock-on:

| Steps after switch | Accuracy |
|--------------------|----------|
| 0-3 | 50.0% |
| 0-5 | 60.0% |
| 0-10 | 80.0% |
| 0-20 | 90.0% |
| Entire block | 98.7% |

### Comparison

| Metric | leak=1.0 | leak=0.6 | Change |
|--------|----------|----------|--------|
| Overall accuracy | 96.8% | 98.7% | +2.0% |
| Chirp accuracy | 88.0% | 96.7% | +8.7% |
| Lock-on @ 3 steps | 75.0% | 50.0% | Slower |
| Lock-on @ 20 steps | 76.2% | 90.0% | Catches up |

### Effect of leak rate on classification

The leaky integrator improves steady-state classification but slows
transition response — a direct tradeoff between accuracy and agility.

**Better chirp-sine separation.** At leak=1.0, chirp is confused with
sine 9% of the time because a chirp starts at a low frequency similar
to sine's 0.08. At leak=0.6, the neurons retain recent frequency history,
so the readout can distinguish chirp's accelerating sweep from sine's
constant frequency. Chirp accuracy jumps from 88% to 97%.

**Slower lock-on after transitions.** When the waveform switches, the
leaky neurons retain state from the previous signal. At leak=1.0, the
reservoir immediately overwrites its state — lock-on is 75% by step 3.
At leak=0.6, old state persists longer — lock-on starts at 50% but
climbs through 80% by step 10 as the new signal gradually overwrites
the old.

**The tradeoff is task-dependent.** For applications where classification
accuracy matters more than transition speed (long blocks, stable signals),
lower leak rates win. For applications where rapid switching is critical
(short blocks, fast-changing inputs), higher leak rates are better. A
multi-timescale reservoir could provide both: fast neurons for quick
lock-on and slow neurons for better steady-state discrimination.

## Things to try

- **Leak rate.** Set `cfg.leak_rate` in the source. The default for this
  example is 0.6. Try 1.0 (instant lock-on, lower accuracy) or 0.3
  (high accuracy, slow transitions).

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
