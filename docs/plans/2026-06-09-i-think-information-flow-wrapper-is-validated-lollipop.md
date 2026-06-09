# Plan: Translate operator.py into C++ as information_flow_causal_map_mask

**Date:** 2026-06-09

---

## Context

`ouxinfo/operator.py` builds a sparse information-flow operator matrix over a masked set of pairs from a multivariate time series. It is slow because:
1. It uses `mpi4py.futures.MPIPoolExecutor` — MPI process spawning and serialization overhead dominate for single-node use.
2. Each pair makes a Python-level `information_flow()` call, paying the GIL and pybind11 overhead on every iteration.

The fix is a C++ wrapper `information_flow_causal_map_mask_wrapper` in `ouxinfo.cpp` that:
- Collects valid pairs from the mask
- Parallelises with OpenMP (same model as `information_flow_causal_map_wrapper`)
- Calls `information_flow<double>()` from `information_flow.hpp` per pair
- Returns a dense `(Nx, Nx)` numpy array

`operator.py` becomes a thin Python shim that calls the C++ function and converts the result to a scipy sparse CSR matrix.

---

## Implementation

### 1. New wrapper in `ouxinfo/ouxinfo.cpp`

Add `information_flow_causal_map_mask_wrapper` **before** the `PYBIND11_MODULE` block, following the exact pattern of `information_flow_causal_map_wrapper` (lines 408–525):

```cpp
py::array_t<double> information_flow_causal_map_mask_wrapper(
    py::array_t<double, py::array::c_style> X_obj,
    py::array_t<bool,   py::array::c_style> mask_obj,
    int tau=1, double dt=1.0, int k=5, int n_threads=1)
{
  py::buffer_info info_X    = X_obj.request();
  py::buffer_info info_mask = mask_obj.request();
  if (info_X.ndim != 2)
    throw std::runtime_error("X must be 2D (Nx, Nt)");
  if (info_mask.ndim != 2)
    throw std::runtime_error("mask must be 2D (Nx, Nx)");
  if (info_X.itemsize != sizeof(double))
    throw std::runtime_error("X must be float64");
  if (info_mask.itemsize != sizeof(bool))
    throw std::runtime_error("mask must be bool");

  int Nx = static_cast<int>(info_X.shape[0]);
  int Nt = static_cast<int>(info_X.shape[1]);
  if (info_mask.shape[0] != Nx || info_mask.shape[1] != Nx)
    throw std::runtime_error("mask must be (Nx, Nx)");

  double* X    = static_cast<double*>(info_X.ptr);
  bool*   mask = static_cast<bool*>(info_mask.ptr);

  // Collect valid pairs from mask (i != j, mask[i,j] == true)
  std::vector<std::pair<int,int>> pairs;
  for (int i = 0; i < Nx; ++i)
    for (int j = 0; j < Nx; ++j)
      if (i != j && mask[i * Nx + j])
        pairs.push_back({i, j});

  // Allocate output (initialised to 0)
  py::array_t<double> result({Nx, Nx});
  double* out = static_cast<double*>(result.request().ptr);
  std::fill(out, out + Nx * Nx, 0.0);

  omp_set_num_threads(n_threads);
  #pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < (int)pairs.size(); ++p) {
    int i = pairs[p].first;
    int j = pairs[p].second;
    double* xi = X + i * Nt;
    double* xj = X + j * Nt;
    // out[i,j] = IF from X_j (source) to X_i (target)
    // Matches information_flow_causal_map convention: IF_map[i,j] = IF from X_j to X_i
    out[i * Nx + j] = information_flow<double>(xj, xi, tau, dt, k, 1, 1, Nt);
  }
  return result;
}
```

Key points:
- Calls `information_flow<double>(xj, xi, ...)` — source=j, target=i — so `out[i,j]` = IF from X_j to X_i.
- This matches the existing `information_flow_causal_map` convention (`IF_map[i,j]` = IF from X_j to X_i).
- Inner `#pragma omp parallel` inside `mi_with_prebuilt_x` runs single-threaded when called from an outer parallel region (OpenMP nested disabled by default) — same behaviour as `information_flow_causal_map_wrapper`.
- No race conditions: each pair writes to a unique `out[i*Nx+j]`.

### 2. Add pybind11 binding in `ouxinfo/ouxinfo.cpp`

Add in the `PYBIND11_MODULE` block (after the `information_flow_causal_map` binding):

```cpp
m.def("information_flow_causal_map_mask", &information_flow_causal_map_mask_wrapper,
      py::arg("X"), py::arg("mask"),
      py::arg("tau")=1, py::arg("dt")=1.0, py::arg("k")=5, py::arg("n_threads")=1,
      R"doc(Build an information-flow operator matrix for a masked set of pairs.

Parameters
----------
X : ndarray of shape (Nx, Nt)
    Multivariate time series, one row per variable. Must be float64.
mask : ndarray of shape (Nx, Nx), dtype bool
    Compute IF only where mask[i, j] is True (and i != j).
tau : int, optional
    Time delay in samples. Default 1.
dt : float, optional
    Physical time step. Default 1.0.
k : int, optional
    Number of nearest neighbors. Default 5.
n_threads : int, optional
    Number of OpenMP threads. Default 1.

Returns
-------
result : ndarray of shape (Nx, Nx)
    result[i, j] = information flow from variable j to variable i.
    Entries where mask[i, j] is False (or i == j) are 0.
    Same convention as information_flow_causal_map: result[i,j] = IF(X_j -> X_i).
)doc");
```

### 3. Replace `ouxinfo/operator.py`

Remove MPI and replace with a thin shim:

```python
import numpy as np
from scipy.sparse import csr_matrix
from ._core import information_flow_causal_map_mask as _information_flow_causal_map_mask_dense


def information_flow_causal_map_mask(x_data, mask, tau=1, dt=1.0, k=5, n_threads=1):
    """
    Build a sparse information-flow operator matrix.

    Parameters
    ----------
    x_data     : ndarray (Nx, Nt)
    mask       : ndarray (Nx, Nx), bool
    tau        : int
    dt         : float
    k          : int
    n_threads  : int

    Returns
    -------
    scipy.sparse.csr_matrix of shape (Nx, Nx)
    """
    x_data = np.ascontiguousarray(x_data, dtype=np.float64)
    mask   = np.ascontiguousarray(mask,   dtype=bool)
    dense  = _information_flow_causal_map_mask_dense(x_data, mask, tau=tau, dt=dt, k=k, n_threads=n_threads)
    return csr_matrix(dense)
```

### 4. Update `ouxinfo/__init__.py`

`operator.py` is not currently imported in `__init__.py`, so no change needed there. Users continue to do `from ouxinfo.operator import information_flow_causal_map_mask` as before.

---

## Files to modify / create

| Action | Path |
|--------|------|
| **Modify** | `ouxinfo/ouxinfo.cpp` — add `information_flow_causal_map_mask_wrapper` + pybind11 binding |
| **Replace** | `ouxinfo/operator.py` — remove MPI, thin shim over `_core.information_flow_causal_map_mask` |

---

## Verification

```bash
# Rebuild
./rebuild.sh

# 1. Correctness: all-True mask must match information_flow_causal_map exactly
python - <<'EOF'
import numpy as np
from ouxinfo import information_flow_causal_map
from ouxinfo.operator import information_flow_causal_map_mask

rng = np.random.default_rng(42)
Nx, Nt = 6, 400
X = rng.standard_normal((Nx, Nt))
tau = 1

# Existing function (per-variable tau array, returns IF_map, Leak_map, dI_map)
tau_arr = np.full(Nx, tau, dtype=np.int32)
IF_existing, _, _ = information_flow_causal_map(X, tau_arr, dt=1.0, k=5, n_threads=1)

# New function with all-True mask
mask = np.ones((Nx, Nx), dtype=bool)
IF_mask_sparse = information_flow_causal_map_mask(X, mask, tau=tau, dt=1.0, k=5, n_threads=1)
IF_mask = IF_mask_sparse.toarray()

# Off-diagonal entries must match (diagonal is NaN in existing, 0 in mask version)
for i in range(Nx):
    for j in range(Nx):
        if i == j:
            continue
        assert abs(IF_existing[i, j] - IF_mask[i, j]) < 1e-10, \
            f"Mismatch at [{i},{j}]: existing={IF_existing[i,j]:.6f}, mask={IF_mask[i,j]:.6f}"
print("OK — all off-diagonal entries match information_flow_causal_map")
EOF

# 2. No regression in existing IF tests
pytest tests/test_IF.py tests/test_IF_regression.py -v
```

Expected: comparison passes to 1e-10 on all off-diagonal entries; all existing tests pass.
