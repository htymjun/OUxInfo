# Plan: OpenMP parallelization of mutual_information.hpp

## Goal

Make ouxinfo's transfer entropy faster than infomeasure at all sample sizes N. Currently ouxinfo TE is slower at N≥5000 (~0.551s vs ~0.380s at N=10000).

## Root cause

`ouxinfo/mutual_information.hpp` has two key functions that run their N-sample query loops serially:

- `mutual_info` (lines 54–76): 1 k-NN + 2 radius searches per sample, serial
- `conditional_mutual_info` (lines 235–265): 1 k-NN + 3 radius searches per sample, serial

Both have `ret_index` / `out_dist` declared before the loop, preventing naive `#pragma omp for`. `shannon_entropy.hpp` and `kullback_leibler_divergence.hpp` already use OpenMP with the correct thread-private pattern.

## Approach

Add `#include <omp.h>` and wrap each loop in `#pragma omp parallel` with thread-private backing vectors, then `#pragma omp for schedule(static)`.

**Thread-safety:** nanoflann `findNeighbors` / `radiusSearch` are `const` (all per-call storage on the stack). `nX[i]` / `nY[i]` / `nZ[i]` etc. writes go to distinct indices per thread — no race.

## Changes

Single file: `ouxinfo/mutual_information.hpp`

1. Add `#include <omp.h>` (after existing includes)
2. `mutual_info`: wrap lines 52–76 in `#pragma omp parallel { ... #pragma omp for schedule(static) ... }`
3. `conditional_mutual_info`: wrap lines 232–265 in the same pattern
4. `mutual_info_Thei`: skip (not on TE hot path)

## Expected speedup

With 4 cores: TE ~0.551s → ~0.14s at N=10000 (well below infomeasure's ~0.380s).

## Verification

```bash
./rebuild.sh
pytest tests/ -v          # full suite must stay green
# speed comparison plot saved to docs/speed_comparison_omp.png
```
