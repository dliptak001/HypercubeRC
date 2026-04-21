# Basic Prediction — Sine Wave Forecasting

## What this example demonstrates

This is the simplest possible end-to-end reservoir computing demo.
A sine wave is fed into the reservoir, and **two readouts** — Ridge
regression and HypercubeCNN — each learn to predict the next value
without ever seeing the input directly. Both run on the same reservoir
dynamics for an apples-to-apples comparison.

If you're new to reservoir computing, start here.

## Conceptual background

In a traditional neural network, you train all the weights. In reservoir
computing, the recurrent network (the "reservoir") has fixed, random weights.
Only the output layer (the "readout") is trained — a single linear regression.

This works because the reservoir transforms the 1-dimensional input into an
N-dimensional state vector that encodes the input's recent history. The readout
just needs to find the right linear combination of those N dimensions to
reconstruct the target.

A sine wave is the easiest test case: the dynamics are perfectly periodic,
so the reservoir state at any point encodes enough history to predict the
next value with near-zero error.

## The pipeline

```
                               ┌──> Output selection ──> Ridge readout ──> Prediction
Input signal ──> Reservoir ────┤      15 vertices          (trained)        sin(0.1(t+1))
  sin(0.1t)      128 neurons   │      (10% of 128)
                  (fixed)      └──> Raw state (all 128) ──> HCNN readout ──> Prediction
                                                             (trained)       sin(0.1(t+1))
```

**Step by step:**

1. **Generate signal** — A sine wave, `sin(0.1t)`. Amplitude stays in [-1, +1],
   which is the reservoir's native input range.

2. **Warmup** — Drive the reservoir for 200 steps without recording. This lets
   the internal state "forget" the arbitrary initial conditions (all zeros) and
   settle into a trajectory that reflects the input history.

3. **Collect** — Drive for 2000 more steps, recording the N-dimensional state
   at each step. This is the training + test data — identical for both readouts.

4. **Ridge path — select outputs** — The `output_fraction` parameter
   (0.1 = 10%) stride-selects 15 of the 128 vertices as raw readout
   features. Ridge fits a linear mapping from these 15 states to the target.

5. **HCNN path — raw state** — The CNN readout uses `output_fraction = 1.0`
   and takes the full 128-vertex raw state as input. No feature selection
   — the convolutional kernels discover their own features on the hypercube.

6. **Train readouts** — Fit both readouts on the same 70% train split.
   Ridge is a closed-form solve; HCNN runs 1100 epochs of Adam with a
   cosine learning-rate schedule.

7. **Evaluate** — Measure R² and NRMSE on the held-out 30% test set for
   both, and print a side-by-side comparison.

## What to expect

### Leak rate = 0.2 (leaky integrator, default)

DIM=7, 128 neurons, leak_rate=0.2. Sine prediction is trivially easy
for both readouts, so both hit near-perfect scores:

- **Ridge** (10% output, 15 raw features): R² effectively 1.0, NRMSE
  in the sub-0.1% range.
- **HCNN** (all 128 vertices, 1 Conv+Pool pair, 1100 epochs): R²
  effectively 1.0, NRMSE within ~2× of Ridge.

On a task this easy the two readouts are indistinguishable — both
find the sine's phase structure from the reservoir state with
effectively zero error. The value of this example is in the pipeline
comparison, not in picking a winner. Harder tasks (NARMA-10,
SignalClassification) are where the readouts pull apart.

### Leak rate = 1.0 (full replacement)

Same configuration with leak_rate=1.0:

Still near-perfect — R² effectively 1.0 for both readouts — but NRMSE
is slightly higher than the leaky integrator case.

### Effect of leak rate on periodic prediction

A sine wave is smooth and perfectly periodic. The leaky integrator
(leak_rate < 1.0) blends old state with new activation at each step:

```
state[v] = (1 - leak) * old_state[v] + leak * tanh(alpha * s)
```

At leak=0.2, neurons retain 80% of their previous state. This acts as
a temporal smoother that benefits periodic signals in two ways:

1. **Phase continuity.** The neuron's state tracks the sine's phase more
   smoothly, reducing step-to-step jitter in the state representation.

2. **Noise suppression.** Any numerical noise from the recurrent dynamics
   is dampened by the averaging effect of the slow update.

The improvement is modest in absolute terms (both are near-perfect) because
sine prediction is trivially easy for a 128-neuron reservoir. The leak rate's
effect is more dramatic on harder tasks — see the StreamingAnomaly and
SignalClassification examples for cases where it substantially changes
detection sensitivity and classification accuracy.

## Things to try

- **Leak rate.** Set `cfg.leak_rate` in the source. The default for this
  example is 0.2. Try 1.0 (no leak), 0.5 (moderate), or 0.05 (very slow)
  and compare NRMSE for both readouts.

- **Output fraction (Ridge only).** Set `ridge_cfg.output_fraction` in
  the source. The default is 0.1 (10% of vertices, giving 15 raw
  features). Try 1.0 for all 128 vertices, or 0.5 for half. The HCNN
  path always uses all vertices.

- **HCNN epochs / learning rate.** The default is 1100 epochs.
  Raising `cnn_cfg.lr_max` above ~0.005 is risky — weights can
  diverge into denormals and the CPU falls off fast-math paths.

- **HCNN layer count.** Leave `cnn_cfg.num_layers = 0` for the
  DIM-auto default (`min(DIM-2, 2)` pairs). Override with a smaller
  count to see how depth affects fit on a trivial signal.

- **Change the signal.** Replace `sin(0.1t)` with a more complex waveform
  (sum of two sines, or a chirp). HCNN's advantage over Ridge grows as
  the target becomes less linear in the reservoir state.

- **Increase the horizon.** Change `horizon = 1` to 5 or 10. Multi-step
  prediction is harder because the reservoir must encode more history.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/BasicPrediction
```
