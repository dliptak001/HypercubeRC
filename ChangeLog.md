# HypercubeRC - Change Log

## v0.2.0 (Apr 23, 2026)

### HypercubeCNN readout (replaces LinearReadout + RidgeRegression)

- Add Readout: topology-native convolutional readout operating directly on hypercube reservoir state
- Auto-sized Conv+Pool stack from DIM with configurable layers (nl), channels (ch), and FLATTEN/GAP head
- Support multi-output regression and multi-class classification (softmax + cross-entropy)
- Add online training API: InitOnline(), TrainOnlineStep(), TrainOnlineBatch(), PredictLiveRaw()
- Add per-DIM frozen baseline configs in Presets.h (DIM 5-10, NARMA-10 tuned)
- Add FLATTEN readout head preserving per-vertex identity end-to-end
- Add output_fraction subsampling: CNN sees a sub-hypercube of the reservoir state
- Add Reservoir::SaveReservoirState() / RestoreReservoirState() for eval checkpointing

### Streaming online training

- Add streaming training mode: one CNN gradient step per reservoir timestep, constant RAM
- Add mini-batch gradient accumulation for streaming training
- Add per-pass evaluation with reservoir state save/restore
- Add linear LR schedule with configurable floor fraction across all passes
- Eliminate hot-path allocations in streaming training loop
- Add docs/TrainingModes.md: batch vs streaming training comparison

### HRCCNN_LM_Text example (character-level language model)

- Add character-level text LM on Tiny Shakespeare (96-token fixed ASCII vocab)
- Streaming online training at DIM 13 (8192 neurons), no states buffer, <50 MiB steady-state RAM
- Multi-pass training with linear LR decay, per-pass eval, autoregressive text sampling
- Binary model serialization with embedded vocab for train/eval/infer workflow
- Eval wraps to corpus start when train+val exceeds corpus length
- Add leak_rate to ReservoirConfig and HRCCNN_LM_Text config plumbing
- Project paused: all configs converge to BPC ~3.05 (reservoir memory bottleneck)
- Add docs/ReservoirMemoryBottleneck.md documenting the ceiling analysis

### Removed components

- Remove LinearReadout and RidgeRegression readout (replaced by Readout)
- Remove TranslationLayer and FeatureMode enum
- Remove Mackey-Glass benchmark and presets
- Remove MemoryCapacity, MemoryCapacityProfile, SeedSurvey, StandaloneESNSweep diagnostics
- Remove CnnSeedSurvey diagnostic (seeds baked into Presets.h)
- Remove HRCCNN_LLM_Math example (superseded by HRCCNN_LM_Text)
- Remove OpenMP dependency (thread pools via HypercubeCNN/ThreadPool.h only)

### Examples and diagnostics

- Merge BasicPrediction and CNNPrediction into single comparison example
- Add HCNN multi-class classification path to SignalClassification
- Add StreamingAnomaly dual-readout variant
- Move CoreSmokeTest to diagnostics/ with HCNN smoke tests (prediction, classification, multi-output)
- Merge BenchmarkSuite and HCNNBenchmarkSuite into unified NARMA-10 suite
- Retune BasicPrediction and SignalClassification epoch counts for HCNN readout

### Documentation

- Rewrite CPP_SDK.md and Python_SDK.md for Readout API
- Add docs/Readout.md: architecture, auto-sizing, training modes, serialization
- Rewrite NARMA10.md, DoesTopologyMatter.md, ScaleInvariance.md with HCNN results
- Move HRCCNN_LM_Text.md to examples/HRCCNN_LM_Text/ alongside its code
- Fix docs/Reservoir.md DIM range: [5, 12] to [5, 16]
- Purge all stale Ridge/TranslationLayer/ReadoutType references from docs
