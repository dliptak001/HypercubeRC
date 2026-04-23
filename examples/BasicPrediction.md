# Basic Prediction — Sine Wave Forecasting

## What this example demonstrates

This is the simplest possible end-to-end reservoir computing demo.
A sine wave is fed into the reservoir, and the HCNN readout learns to
predict the next value without ever seeing the input directly.

If you're new to reservoir computing, start here.

## Conceptual background

In a traditional neural network, you train all the weights. In reservoir
computing, the recurrent network (the "reservoir") has fixed, random weights.
Only the output layer (the "readout") is trained.

This works because the reservoir transforms the 1-dimensional input into an
N-dimensional state vector that encodes the input's recent history. The readout
finds the right mapping from those N dimensions to the target.

A sine wave is the easiest test case: the dynamics are perfectly periodic,
so the reservoir state at any point encodes enough history to predict the
next value with near-zero error.

## The pipeline

```
Input signal ──> Reservoir ──> HCNNReadout ──> Prediction
  sin(0.1t)      128 neurons     (trained)      sin(0.1(t+1))
                  (fixed)
```

**Step by step:**

1. **Generate signal** — A sine wave, `sin(0.1t)`. Amplitude stays in [-1, +1],
   which is the reservoir's native input range.

2. **Warmup** — Drive the reservoir for 200 steps without recording. This lets
   the internal state settle into a trajectory that reflects the input history.

3. **Collect** — Drive for 2000 more steps, recording the N-dimensional state
   at each step. This is the training + test data.

4. **Train** — The HCNN readout takes the full 128-vertex raw state as input.
   Convolutional kernels discover features on the hypercube topology. Trained
   with 1100 epochs of Adam with a cosine learning-rate schedule.

5. **Evaluate** — Measure R² and NRMSE on the held-out 30% test set.

## What to expect

### Leak rate = 0.2 (leaky integrator)

DIM=7, 128 neurons, leak_rate=0.2. Sine prediction is trivially easy:

- **HCNN** (all 128 vertices, 1 Conv+Pool pair, 1100 epochs): R²
  effectively 1.0, NRMSE in the sub-0.1% range.

The value of this example is in demonstrating the pipeline, not
stressing the readout. Harder tasks (NARMA-10, SignalClassification)
are where the architecture shows its capacity.

### Leak rate = 1.0 (full replacement)

Same configuration with leak_rate=1.0:

Still near-perfect — R² effectively 1.0 — but NRMSE is slightly higher
than the leaky integrator case.

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
SignalClassification examples.

## Things to try

- **Leak rate.** Set `cfg.leak_rate` in the source. The default for this
  example is 0.2. Try 1.0 (no leak), 0.5 (moderate), or 0.05 (very slow)
  and compare NRMSE.

- **HCNN epochs / learning rate.** The default is 1100 epochs.
  Raising `cnn_cfg.lr_max` above ~0.005 is risky — weights can
  diverge into denormals and the CPU falls off fast-math paths.

- **HCNN layer count.** The preset uses `num_layers = 1` (one Conv+Pool
  pair). Set `cnn_cfg.num_layers = 0` for auto-sizing (`min(DIM-2, 2)`
  pairs) or increase to see how depth affects fit on a trivial signal.

- **Change the signal.** Replace `sin(0.1t)` with a more complex waveform
  (sum of two sines, or a chirp). HCNN's advantage grows as the target
  becomes more nonlinear in the reservoir state.

- **Increase the horizon.** Change `horizon = 1` to 5 or 10. Multi-step
  prediction is harder because the reservoir must encode more history.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/BasicPrediction
```
