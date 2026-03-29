# Basic Prediction — Sine Wave Forecasting

## What this example demonstrates

This is the simplest possible end-to-end reservoir computing demo.
A sine wave is fed into the reservoir, and a linear readout learns to
predict the next value — without ever seeing the input directly.
The readout works entirely from the reservoir's internal state.

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
Input signal ──> Reservoir ──> State features ──> Readout ──> Prediction
  sin(0.1t)      128 neurons    128 or 320        Ridge       sin(0.1(t+1))
                  (fixed)        floats/step       (trained)
```

**Step by step:**

1. **Generate signal** — A sine wave, `sin(0.1t)`. Amplitude stays in [-1, +1],
   which is the reservoir's native input range.

2. **Warmup** — Drive the reservoir for 200 steps without recording. This lets
   the internal state "forget" the arbitrary initial conditions (all zeros) and
   settle into a trajectory that reflects the input history.

3. **Collect** — Drive for 2000 more steps, recording the N-dimensional state
   at each step. This is the training + test data.

4. **Extract features** — Either use the raw N states, or apply the translation
   layer to get 2.5N features (x, x², x*x'). Translation is overkill for a
   sine wave but demonstrates the pipeline.

5. **Train readout** — Fit a Ridge or Linear readout on 70% of the data,
   mapping state features to the signal value one step in the future.

6. **Evaluate** — Measure R² and NRMSE on the held-out 30% test set.

## What to expect

With DIM=7 (128 neurons) and Ridge regression:

| Metric | Value | Meaning |
|--------|-------|---------|
| R² | 1.000000 | Effectively perfect fit |
| NRMSE | ~0.0004 | Sub-0.1% error |

Prediction errors are in the 4th decimal place — the reservoir encodes enough
of the sine's history for near-exact one-step prediction.

## Things to try

- **Switch readout type.** Change `ReadoutType::Ridge` to `ReadoutType::Linear`
  in the ESN constructor. Ridge gives slightly better results on this easy task;
  the difference is more dramatic on harder benchmarks.

- **Change the signal.** Replace `sin(0.1t)` with a more complex waveform
  (e.g., sum of two sines, or a chirp). The reservoir handles these too, but
  prediction error will increase.

- **Increase the horizon.** Change `horizon = 1` to 5 or 10. Multi-step
  prediction is harder because the reservoir must encode more history.

- **Try raw vs. translation features.** Pass `raw` or `translation` as a
  command-line argument. For a simple sine, raw features are sufficient.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/BasicPrediction              # default: raw features, Ridge readout
./build/BasicPrediction raw          # explicit raw
./build/BasicPrediction translation  # translation 2.5N features
```
