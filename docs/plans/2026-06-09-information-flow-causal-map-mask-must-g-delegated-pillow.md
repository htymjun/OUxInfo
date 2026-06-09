# Plan: information_flow_causal_map_mask — accept tau array

## Context

`information_flow_causal_map_mask_wrapper` currently accepts only a scalar `int tau`, while the non-mask sibling `information_flow_causal_map_wrapper` already supports a per-variable `tau` array (shape `(N,)`). The user wants the mask variant to have the same capability so that different time delays can be assigned to different variables when using a mask.

The only structural difference between the two variants is the mask: the mask variant skips pairs where `mask[i,j] == false`, returning a sparse (N,N) array of IF values. The tau-array logic is identical — each source variable `j` uses `tau_arr[j]`.

## Files to modify

[ouxinfo/ouxinfo.cpp](ouxinfo/ouxinfo.cpp) — two locations:

---

## Change 1: `information_flow_causal_map_mask_wrapper` signature and loop (lines 528–588)

**Current signature:**
```cpp
py::array_t<double> information_flow_causal_map_mask_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::object mask_obj,
  int tau=1, double dt=1.0, int k=5, int n_threads=1)
```

**New signature:**
```cpp
py::array_t<double> information_flow_causal_map_mask_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::object mask_obj,
  py::array_t<int, py::array::c_style> tau_obj,
  double dt=1.0, int k=5, int n_threads=1)
```

**Add after `Nx`/`Nt`/`X` extraction:**
```cpp
py::buffer_info info_tau = tau_obj.request();
if (info_tau.ndim != 1 || info_tau.shape[0] != Nx)
  throw std::runtime_error("tau must be 1D with length Nx");
if (info_tau.itemsize != sizeof(int))
  throw std::runtime_error("tau must be int32");
int* tau_arr = static_cast<int*>(info_tau.ptr);
```

**Change the inner loop call** (line 585) from:
```cpp
out[i * Nx + j] = information_flow<double>(xj, xi, tau, dt, k, 1, 1, Nt);
```
to:
```cpp
out[i * Nx + j] = information_flow<double>(xj, xi, tau_arr[j], dt, k, 1, 1, Nt);
```

Convention: source is `j`, so `tau_arr[j]` is used — consistent with the full wrapper where `IF[i,j]` (X_j→X_i) uses `tau_j`.

---

## Change 2: Dispatcher `mask` branch (lines 614–619)

**Current:**
```cpp
} else {
  // Mask provided: masked computation (returns single array)
  if (py::isinstance<py::array>(tau_obj))
    throw std::runtime_error("tau must be a scalar int when mask is provided");
  int tau = tau_obj.cast<int>();
  return information_flow_causal_map_mask_wrapper(X_obj, mask_obj, tau, dt, k, n_threads);
}
```

**New:** reuse the same scalar→array broadcast already done for the no-mask branch:
```cpp
} else {
  // Mask provided: masked computation (returns single array)
  // tau may be scalar int or int32 array; broadcast scalar to per-variable array
  py::array_t<int, py::array::c_style> tau_arr;
  if (py::isinstance<py::array>(tau_obj)) {
    tau_arr = tau_obj.cast<py::array_t<int, py::array::c_style>>();
  } else {
    int tau_val = tau_obj.cast<int>();
    int Nx = static_cast<int>(X_obj.request().shape[0]);
    tau_arr = py::array_t<int>({Nx});
    auto buf = tau_arr.mutable_unchecked<1>();
    for (int i = 0; i < Nx; ++i) buf(i) = tau_val;
  }
  return information_flow_causal_map_mask_wrapper(X_obj, mask_obj, tau_arr, dt, k, n_threads);
}
```

---

## Verification

After rebuilding (`./rebuild.sh`):

```bash
pytest tests/test_IF_mask.py   # existing mask tests still pass
```

Manual smoke test (scalar tau still works):
```python
import numpy as np
from ouxinfo import information_flow_causal_map
X = np.random.randn(3, 500)
mask = np.ones((3,3), dtype=bool); np.fill_diagonal(mask, False)
# scalar tau (broadcast)
IF1 = information_flow_causal_map(X, tau=1, mask=mask)
# tau array
taus = np.array([1, 2, 1], dtype=np.int32)
IF2 = information_flow_causal_map(X, tau=taus, mask=mask)
print(IF1.shape, IF2.shape)  # both (3, 3)
```
