# HypercubeRC

[![Build wheels](https://github.com/dliptak001/HypercubeRC/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeRC/actions/workflows/wheels.yml)

Python bindings for reservoir computing on Boolean hypercube graphs.

The reservoir topology is a Boolean hypercube of dimension DIM, giving
N = 2^DIM neurons (DIM 5-12, i.e. 32 to 4096 neurons). All connectivity
is defined by XOR operations on vertex indices -- no adjacency list stored.
Scale-invariant hyperparameters: the same SR and input_scaling work at every DIM.

## Installation

```bash
pip install hypercube-rc
```

Pre-built wheels for Python 3.10-3.13 on Windows (x64), Linux (x86_64,
aarch64), and macOS (x86_64, arm64). No compiler required.

## Quick Start

```python
import numpy as np
import hypercube_rc as hrc

# One-step-ahead sine prediction
signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
esn = hrc.ESN(dim=7, seed=42)
esn.fit(signal, warmup=200)
print(f"R2 = {esn.r2():.6f}")      # R2 = 0.999999
print(f"NRMSE = {esn.nrmse():.6f}") # NRMSE = 0.000129
```

## Features

- **Simple API** -- `fit()` handles warmup, run, and train in one call
- **DIM 5-12** -- 32 to 4096 neurons, scale-invariant defaults
- **HCNN readout** -- learned convolutional readout on raw reservoir state
- **Multi-input** -- multiple input channels via stride-interleaved driving
- **Streaming mode** -- online training for real-time applications
- **Model persistence** -- pickle, save/load to disk

## Documentation

Full API reference: [docs/Python_SDK.md](https://github.com/dliptak001/HypercubeRC/blob/master/docs/Python_SDK.md)

Project repository: [github.com/dliptak001/HypercubeRC](https://github.com/dliptak001/HypercubeRC)
