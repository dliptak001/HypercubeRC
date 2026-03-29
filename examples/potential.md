# Potential Examples

Candidates for additional example programs. Each demonstrates a distinct capability
not covered by BasicPrediction (regression) or StreamingAnomaly (anomaly detection).

Ranked by impact — highest first.

## 1. Signal Classification

Classify waveform types (sine, square, triangle, chirp) from reservoir states. This is
the only use case that exercises `Predict()` (thresholded classification) and `Accuracy()`
rather than `PredictRaw()` (regression). Could do pairwise or multi-class via one-vs-rest.
Clean confusion-matrix output. Shows the reservoir as a feature extractor for pattern
recognition, not just prediction.

## 2. Multi-Horizon Prediction

Mackey-Glass chaotic series predicted at horizons h=1, 2, 5, 10, 20, 50. One reservoir
drive, multiple readouts trained on different lookaheads. Produces a clean table showing
NRMSE degrading with horizon — the classic demonstration of how reservoir memory capacity
limits prediction range. Directly connects MC benchmark numbers to practical prediction
limits.

## 3. Frequency Decomposition

Feed a mixed signal (3 harmonics at different frequencies), train separate readouts to
extract each component individually. Shows the reservoir simultaneously represents all
frequency components — each readout pulls out a different one from the same state vector.
Intuitive demonstration that the N-dimensional state is a rich representation, not just
a single "prediction."

## 4. Nonlinear Channel Equalization

A classic RC benchmark from telecom. A binary signal passes through a nonlinear channel
(polynomial distortion + noise), and the reservoir learns the inverse mapping to recover
the original bits. Reports bit error rate (BER) at different SNR levels. Practical
application, clean tabular output, and exercises the classification pathway on a
real-world problem.
