# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

This is a CLion C++ project. cmake/g++ are NOT on PATH; they are bundled with CLion.

**Build (Release):**
```bash
powershell.exe -File - <<'PS1'
$cmake = 'C:\Program Files\JetBrains\CLion 2024.3.2\bin\cmake\win\x64\bin\cmake.exe'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& $cmake --build C:\CLion\HypercubeRC\cmake-build-release 2>&1
PS1
```

**Run any executable** (MinGW DLLs needed on PATH):
```bash
powershell.exe -File - <<'PS1'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& "C:\CLion\HypercubeRC\cmake-build-release\HypercubeRC.exe" 2>&1
PS1
```

Replace `HypercubeRC.exe` with any target: `BasicPrediction.exe`, `SignalClassification.exe`, `StreamingAnomaly.exe`, `HRCCNN_LM_Text.exe`, `CoreSmokeTest.exe`.

**Critical rules:**
- NEVER reconfigure cmake-build-* directories (`cmake -B` with `-G` flags). CLion owns them.
- NEVER use bare bash to invoke g++ or cmake. Errors are silently swallowed. Always use the PowerShell heredoc pattern above.
- Prefer Release builds (Debug has different float behavior with `-ffast-math`).
- External dependency: HypercubeCNN (sibling project at `../HypercubeCNN`, linked as static library).

## Architecture

HypercubeRC is a reservoir computing library where the reservoir topology is a Boolean hypercube graph (N = 2^DIM neurons, DIM 5-16).

### Pipeline

1. **Reservoir** (`Reservoir.h/.cpp`) — Fixed recurrent network. N neurons sit on hypercube vertices; connectivity is computed inline via XOR masks (no adjacency storage). Each neuron receives from 2*DIM-2 neighbors (shell masks + nearest-neighbor single-bit flips). Weights initialized randomly then rescaled to target spectral radius.

2. **Readout** (`Readout.h/.cpp`) — HypercubeCNN-based learned readout. Operates directly on raw reservoir state. Auto-sized Conv→Pool stack from DIM, supports multi-output regression and multi-class classification. See `docs/Readout.md` for full details.

**ESN** (`ESN.h/cpp`) — Unified pipeline wrapper. Reservoir → Readout. Provides Train(), PredictRaw(), R2(), NRMSE(), Accuracy(), NumOutputs(). Supports multi-output prediction and multi-class classification.

### Key design properties

- **XOR addressing**: Neighbor of vertex v is `v XOR mask` — O(1) lookup, zero storage, trivially parallelizable
- **Scale-invariant defaults**: SR=0.90 and input_scaling=0.02 are optimal across all DIM values (vertex-transitive topology property). No per-size re-tuning needed.
- **Template on DIM**: Core classes are `template<size_t DIM>` with N = 2^DIM computed at compile time
### Diagnostics (`diagnostics/`)

- `BenchmarkSuite.h` — Unified NARMA-10 HCNN benchmark (library supports DIM 5-16)
- `NARMA10.h` — Standard nonlinear benchmark
- `StateRank.h` — Effective dimensionality analysis

### Examples (`examples/`)

`BasicPrediction`, `SignalClassification`, `StreamingAnomaly` — each with a companion `.md` walkthrough. `HRCCNN_LM_Text` (character-level text LM on Tiny Shakespeare, DIM 12) — design doc in `docs/`.

## Code Conventions

- C++23 with `-Wall -Wextra`
- Core pipeline implementations in `.cpp` files: `Reservoir.cpp`, `ESN.cpp`, `Readout.cpp`
- Static library `HypercubeRCCore` built from core sources; executables link against it
- Detailed documentation in `docs/` covering architecture rationale and benchmark results
