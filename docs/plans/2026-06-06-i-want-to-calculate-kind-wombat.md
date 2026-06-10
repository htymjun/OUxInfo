# Context

Optimize the information flow calculation. The existing code has two entry points:

1. **C++ single-node**: `information_flow_causal_map_wrapper` in `ouxinfo/ouxinfo.cpp` — used by all single-machine callers
2. **Python MPI distributed**: `information_flow_causal_map_mpi` in `ouxinfo/distributed.py` — added for HPC multi-node

Both have redundant work that can be eliminated without changing any algorithm math.

---

## Root Cause of Redundancy

Each call to `mutual_info(X, Y, ...)` (in `ouxinfo/mutual_information.hpp:102`) builds **three structures from scratch**:
1. Joint XY k-d tree (always required — unique per pair)
2. X marginal: `SortedArray1D` (1D) or k-d tree (>1D)
3. Y marginal: `SortedArray1D` (1D) or k-d tree (>1D)

Building each sorted array or tree is O(Nt log Nt). For N=100 variables, the causal map makes **~24,750 MI calls**, building **~74,250 structures** — but many marginals are redundant.

### Redundancy in `distributed.py` (Python, easy)

For each unique pair (j < i) the current code calls `information_flow(Xi, Xj)` and `information_flow(Xj, Xi)` as black-box wrappers, then computes dI separately:

```
information_flow(Xi, Xj, tau_i)   →  MI(Xi, Xj[tau_i:])  +  MI(Xi, Xj)    ← call A, call B
information_flow(Xj, Xi, tau_j)   →  MI(Xj, Xi[tau_j:])  +  MI(Xj, Xi)    ← call C, call D
mutual_info(Xi[:Neff_i], Xj[:Neff_i])  ←  ALWAYS equals call B! (call E)
mutual_info(Xi[tau_i:],  Xj[tau_i:])   ←  call F
```

**Call E = Call B** unconditionally (same data). **Call D = Call B** when `tau_i == tau_j` (common case: uniform tau). This wastes 1–2 MI calls per pair.

### Redundancy in `ouxinfo.cpp` (C++, higher impact)

Within the j-loop, for each target j the inner loop iterates over all sources i. The unshifted `Xj` is the **same for all N-1 inner iterations**. Yet `mutual_info(&Xi, &Xj, ...)` (call B) rebuilds the `Xj` sorted array from scratch for every i. Similarly, `Xi` sorted array is rebuilt twice within each i-iteration (once in call A, once in call B).

If all tau values are uniform (the standard shell-model setup), the **shifted** `Xj[tau:]` sorted array is also the same for all i.

**For N=100, uniform tau, 1D data:**  
Marginals that are redundantly rebuilt: **~19,800 sorted arrays** (≈67% of all builds) can be eliminated by precomputing once per variable.

---

## Implementation Plan

### Part 1 — Python: rewrite `information_flow_causal_map_mpi` in `ouxinfo/distributed.py`

Inline the MI calls instead of delegating to the `information_flow` wrapper. This eliminates call E (always) and call D (when tau uniform):

```python
for j, i in local_pairs:          # j < i
    Xj, Xi = _ensure_2d(X[j]), _ensure_2d(X[i])
    tau_i, tau_j = int(tau[i]), int(tau[j])
    Neff_i, Neff_j = Nt - tau_i, Nt - tau_j

    I_base_i = _core.mutual_info(Xi[:Neff_i], Xj[:Neff_i], k=k)   # was calls B+E
    Ilag_ji  = _core.mutual_info(Xi[:Neff_i], Xj[tau_i:], k=k)    # call A
    Ilag_ij  = _core.mutual_info(Xj[:Neff_j], Xi[tau_j:], k=k)    # call C
    I_lag_xy = _core.mutual_info(Xi[tau_i:],  Xj[tau_i:], k=k)    # call F

    I_base_j = I_base_i if tau_i == tau_j else \
               _core.mutual_info(Xj[:Neff_j], Xi[:Neff_j], k=k)   # call D only if needed

    local_IF[(j, i)] = float((Ilag_ji - I_base_i) / dt)
    local_IF[(i, j)] = float((Ilag_ij - I_base_j) / dt)
    local_dI[(j, i)] = float((I_lag_xy - I_base_i) / dt)
```

**Result:** 6 → 4 MI calls per unique pair (uniform tau), or 6 → 5 (non-uniform). ~17–33% fewer C++ calls in the MPI path.

---

### Part 2 — C++: precomputed marginals for `information_flow_causal_map_wrapper`

#### Step A: Add `mutual_info_prebuilt` to `ouxinfo/mutual_information.hpp`

New template function that accepts already-built marginal structures (null = build fresh):

```cpp
template<typename T>
T mutual_info_prebuilt(
    T* X, T* Y, int k, int dx, int dy, int N,
    const SortedArray1D<T>* sa_X,          // nullptr → build from X
    const SortedArray1D<T>* sa_Y           // nullptr → build from Y
)
```

Implementation is identical to `mutual_info` except the marginal-build blocks are skipped when the pre-built pointer is non-null. The joint XY tree is always rebuilt (it depends on both variables).

#### Step B: Modify `information_flow_causal_map_wrapper` in `ouxinfo/ouxinfo.cpp`

Before the OpenMP parallel region, detect the fast path and precompute:

```cpp
bool fast_path = (dx == 1);
int  tau_uniform = tau_arr[0];
for (int i = 1; i < N && fast_path; ++i)
    if (tau_arr[i] != tau_uniform) fast_path = false;

// Precompute per-variable sorted arrays (fast path: uniform tau, 1D data)
std::vector<SortedArray1D<double>> sa_base(N), sa_shift(N);
if (fast_path) {
    int Neff = Nt - tau_uniform;
    for (int v = 0; v < N; ++v) {
        sa_base[v].build(X + v*Nt, Neff);                   // Xv[:Neff]
        sa_shift[v].build(X + v*Nt + tau_uniform, Neff);    // Xv[tau:]
    }
}
```

Inside the parallel loop, replace each `mutual_info` call with `mutual_info_prebuilt`, passing the appropriate precomputed pointers:

| Current call | Precomputed X-marginal | Precomputed Y-marginal |
|---|---|---|
| `MI(Xi, Xj_shifted)` — call A | `sa_base[i]` | `sa_shift[j]` |
| `MI(Xi, Xj)` — call B | `sa_base[i]` | `sa_base[j]` |
| `MI(Xi_shifted, Xj_shifted)` — call C | `sa_shift[i]` | `sa_shift[j]` |

On the non-fast path (multi-dim or non-uniform tau), all three pointer args are `nullptr` → falls back to exact current behavior.

**Result:** Saves 2 sorted array builds per MI call × 2.5 calls per pair = 5 builds saved per pair. For N=100: saves ~49,500 of the original ~74,250 structure builds (**~67% reduction**). Estimated wall-clock speedup for 1D uniform-tau data: **~30–40%**.

---

## Files to Modify

| File | Change |
|---|---|
| `ouxinfo/distributed.py` | Rewrite inner loop of `information_flow_causal_map_mpi` — inline MI calls |
| `ouxinfo/mutual_information.hpp` | Add `mutual_info_prebuilt` template function |
| `ouxinfo/ouxinfo.cpp` | Modify `information_flow_causal_map_wrapper` to precompute and pass sorted arrays |

`transfer_entropy_causal_map` and all other functions are **unchanged**.

---

## Verification

### Step 0: Baseline speed measurement (run BEFORE any C++ changes)

Record wall time for the current single-node implementation to have a reference:
```bash
python -c "
import time, numpy as np
from ouxinfo import information_flow_causal_map
X   = np.random.default_rng(0).standard_normal((30, 2000)).astype(np.float64)
tau = np.ones(30, dtype=np.int32)
# warm-up
information_flow_causal_map(X[:4], tau[:4], n_threads=1)
# timed run
t0 = time.perf_counter()
information_flow_causal_map(X, tau, n_threads=1)
print(f'BEFORE: {time.perf_counter()-t0:.3f}s  (N=30, Nt=2000, n_threads=1)')
"
```

### Step 1: Python correctness (MPI)

Single-rank equivalence with the original:
```bash
mpirun -n 1 python -c "
import numpy as np
from ouxinfo import information_flow_causal_map, information_flow_causal_map_mpi
rng = np.random.default_rng(0)
X = rng.standard_normal((8, 300)); tau = np.ones(8, dtype=np.int32)
ref = information_flow_causal_map(X, tau)
mpi = information_flow_causal_map_mpi(X, tau)
for a, b in zip(ref, mpi): assert np.allclose(a, b, equal_nan=True), 'FAIL'
print('OK')
"
```

### Step 2: C++ correctness — rebuild and run test suite
```bash
./rebuild.sh && pytest tests/
```

### Step 3: Post-modification speed measurement (run AFTER C++ changes + rebuild)

Re-run the same benchmark and compare against the Step 0 baseline:
```bash
python -c "
import time, numpy as np
from ouxinfo import information_flow_causal_map
X   = np.random.default_rng(0).standard_normal((30, 2000)).astype(np.float64)
tau = np.ones(30, dtype=np.int32)
information_flow_causal_map(X[:4], tau[:4], n_threads=1)   # warm-up
t0 = time.perf_counter()
information_flow_causal_map(X, tau, n_threads=1)
print(f'AFTER:  {time.perf_counter()-t0:.3f}s  (N=30, Nt=2000, n_threads=1)')
"
```

Expected: AFTER time should be 25–40% lower than BEFORE for 1D uniform-tau data.
