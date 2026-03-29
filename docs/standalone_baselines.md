# Standalone ESN Baselines

## Raw Features (N)

Per-DIM default SR and input scaling, jointly optimized on raw N-dim readout.
3-seed average {42, 1042, 2042}, LinearReadout, warmup 200 (DIM<8) / 500 (DIM>=8),
collect 18*N.

| DIM |   N |  SR |  inp |   MG h=1 |   MG h=5 |  MG h=10 | NARMA-10 |    MC |
|-----|-----|-----|------|----------|----------|----------|----------|-------|
|   4 |  16 | 0.95 | 0.05 |  0.00976 |  0.21164 |  0.52797 |   0.8441 |  9.18 |
|   5 |  32 | 0.80 | 0.10 |  0.01028 |  0.20618 |  0.55548 |   0.5463 | 13.01 |
|   6 |  64 | 0.90 | 0.05 |  0.00842 |  0.18937 |  0.53034 |   0.4203 | 16.70 |
|   7 | 128 | 0.88 | 0.03 |  0.00591 |  0.16392 |  0.47898 |   0.3948 | 24.69 |
|   8 | 256 | 0.88 | 0.02 |  0.00685 |  0.16754 |  0.48212 |   0.4035 | 26.51 |

---

## Translation Layer Features (2.5N)

Per-DIM default SR and input scaling, jointly optimized on translated 2.5N readout.
Same methodology as raw: 3-seed average, LinearReadout, warmup/collect as above.

| DIM |   N |   NF |  SR |  inp |   MG h=1 |   MG h=5 |  MG h=10 | NARMA-10 |    MC |
|-----|-----|------|-----|------|----------|----------|----------|----------|-------|
|   4 |  16 |   40 | 0.88 | 0.02 |  0.01179 |  0.21930 |  0.44814 |   0.9079 |  7.78 |
|   5 |  32 |   80 | 0.80 | 0.04 |  0.00933 |  0.11255 |  0.21024 |   0.5319 | 11.81 |
|   6 |  64 |  160 | 0.92 | 0.02 |  0.00853 |  0.10227 |  0.17471 |   0.2701 | 15.35 |
|   7 | 128 |  320 | 0.92 | 0.04 |  0.00519 |  0.07629 |  0.12943 |   0.1897 | 25.40 |
|   8 | 256 |  640 | 0.95 | 0.02 |  0.00564 |  0.06293 |  0.08753 |   0.1338 | 30.92 |

---

## Translation vs Raw (% change)

Negative = translation is better (lower NRMSE or higher MC).

| DIM |   N |  MG h=1 |  MG h=5 | MG h=10 | NARMA-10 |    MC |
|-----|-----|---------|---------|---------|----------|-------|
|   4 |  16 |  +20.8% |  +3.6%  | -15.1%  |   +7.6%  | -15.3% |
|   5 |  32 |  -9.2%  | -45.4%  | -62.2%  |   -2.6%  |  -9.2% |
|   6 |  64 |  +1.3%  | -46.0%  | -67.1%  |  -35.7%  |  -8.1% |
|   7 | 128 | -12.2%  | -53.5%  | -73.0%  |  -52.0%  |  +2.9% |
|   8 | 256 | -17.7%  | -62.4%  | -81.8%  |  -66.8%  | +16.6% |

---

## Optimization method

Coarse-to-fine grid sweep using `sweeps/StandaloneESNSweep.cpp`
(`USE_TRANSLATION = false` for raw, `true` for translation).
Round 1 covers wide SR and input_scaling ranges, rounds 2-3 zoom into
winners until all optima are interior.

Balanced defaults chosen to be strong across all tasks (MG, NARMA, MC),
not optimized for any single benchmark.
