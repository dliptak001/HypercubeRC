# Hypercube Neighbor Masks

Two families of XOR masks define which vertices each vertex reads from.
Both are computed inline from the loop index — no adjacency storage.

## Mask Families

### Shell (Recurrent) — `ShellMask(i) = (1 << (i+1)) - 1`

Cumulative low-bit selectors. Vertex v reads from `v ^ mask`, which gives
neighbors within growing Hamming-distance balls from the low-bit end.

**Hamming distance:** 1, 2, 3, ..., DIM (monotonically increasing).

### Nearest (Recurrent) — `NearestMask(i) = 1 << i`

Single-bit flips. Each connection toggles exactly one dimension.

**Hamming distance:** Always 1 (immediate neighbors in each dimension).

## Topological Relationships

The two families cover different parts of the Hamming distance spectrum:

| Family   | Formula              | Hamming dist (from v) | Direction       |
|----------|----------------------|-----------------------|-----------------|
| Nearest  | `1 << i`             | 1 (constant)          | Immediate       |
| Shell    | `(1 << (i+1)) - 1`  | 1 to DIM              | Local -> global |

**Nearest** provides uniform single-hop connectivity across all DIM dimensions.
**Shell** provides progressively longer-range connections that mix more bits.

## Connection Counts

Each vertex has 2*DIM recurrent connections (DIM Shell + DIM Nearest).
Input enters through a separate dense W_in projection (additive to `vtx_output_`),
not through the weighted sum.

## Value Overlap

Shell and Nearest share exactly one mask value across all DIM:

| Pair            | Shared values | Which value                        |
|-----------------|---------------|------------------------------------|
| Shell / Nearest | 1             | ShellMask(0) = NearestMask(0) = 1 |

The first shell mask and the first nearest mask both equal 1 (flip bit 0).
All other values are distinct.

## Full Mask Tables

### DIM=5  (N=32, MASK=31)

| i | Shell | Nearest |
|---|-------|---------|
| 0 |   1   |   1     |
| 1 |   3   |   2     |
| 2 |   7   |   4     |
| 3 |  15   |   8     |
| 4 |  31   |  16     |


### DIM=6  (N=64, MASK=63)

| i | Shell | Nearest |
|---|-------|---------|
| 0 |   1   |   1     |
| 1 |   3   |   2     |
| 2 |   7   |   4     |
| 3 |  15   |   8     |
| 4 |  31   |  16     |
| 5 |  63   |  32     |


### DIM=7  (N=128, MASK=127)

| i | Shell | Nearest |
|---|-------|---------|
| 0 |   1   |   1     |
| 1 |   3   |   2     |
| 2 |   7   |   4     |
| 3 |  15   |   8     |
| 4 |  31   |  16     |
| 5 |  63   |  32     |
| 6 | 127   |  64     |


### DIM=8  (N=256, MASK=255)

| i | Shell | Nearest |
|---|-------|---------|
| 0 |   1   |   1     |
| 1 |   3   |   2     |
| 2 |   7   |   4     |
| 3 |  15   |   8     |
| 4 |  31   |  16     |
| 5 |  63   |  32     |
| 6 | 127   |  64     |
| 7 | 255   | 128     |


### DIM=9  (N=512, MASK=511)

| i | Shell | Nearest |
|---|-------|---------|
| 0 |   1   |   1     |
| 1 |   3   |   2     |
| 2 |   7   |   4     |
| 3 |  15   |   8     |
| 4 |  31   |  16     |
| 5 |  63   |  32     |
| 6 | 127   |  64     |
| 7 | 255   | 128     |
| 8 | 511   | 256     |


### DIM=10  (N=1024, MASK=1023)

| i | Shell  | Nearest |
|---|--------|---------|
| 0 |    1   |    1    |
| 1 |    3   |    2    |
| 2 |    7   |    4    |
| 3 |   15   |    8    |
| 4 |   31   |   16    |
| 5 |   63   |   32    |
| 6 |  127   |   64    |
| 7 |  255   |  128    |
| 8 |  511   |  256    |
| 9 | 1023   |  512    |

