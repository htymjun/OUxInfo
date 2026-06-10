# Plan: IF-only causal map + move to information_flow.cpp

## Context

`information_flow_causal_map` always computes three outputs `(IF, Leak, dI)`. Computing `dI` costs one extra MI call per pair (~20% overhead), and `Leak` is derived algebraically. When only `IF` is needed, this overhead is wasteful. The user wants a `full=True` flag on the existing function so callers can opt out of `Leak`/`dI`. Separately, all four C++ implementation functions should live in a new `information_flow.cpp` rather than `ouxinfo.cpp`.

## Files to change

| File | Action |
|------|--------|
| `ouxinfo/information_flow.cpp` | **Create** — the four wrapper functions + one unified dispatcher |
| `ouxinfo/ouxinfo.cpp` | Remove moved functions; swap include; add `full` arg to `m.def` |

`information_flow.hpp` is untouched (holds pure-C++ `mi_with_prebuilt_x` and `information_flow` templates).

---

## information_flow.cpp structure

```cpp
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "information_flow.hpp"

namespace py = pybind11;
```

### Four implementation functions

**1. `information_flow_causal_map_wrapper`** (existing, no mask)
`(X_obj, tau_obj, dt, k, n_threads) → py::tuple(IF, Leak, dI)`
Moved verbatim from `ouxinfo.cpp`.

**2. `information_flow_causal_map_mask_wrapper`** (existing, with mask)
`(X_obj, mask_obj, tau_obj, dt, k, n_threads) → py::tuple(IF, Leak, dI)`
Moved verbatim from `ouxinfo.cpp`.

**3. `information_flow_only_causal_map_wrapper`** (new, no mask)
`(X_obj, tau_obj, dt, k, n_threads) → py::array_t<double> IF_map`
Same as function 1 but:
- Allocates only `IF_map` (no `dI_map`, no `Leak_map`).
- Omits the `dI` MI call (`mutual_info(&xi_s, &xj_s, ...)` block).
- Omits the Leak post-pass.
- Returns `IF_map` directly (not a tuple).

**4. `information_flow_only_causal_map_mask_wrapper`** (new, with mask)
`(X_obj, mask_obj, tau_obj, dt, k, n_threads) → py::array_t<double> IF_map`
Same as function 2 but allocates/returns only `IF_map`, skips dI and Leak.

### One unified dispatcher

```cpp
py::object information_flow_causal_map_dispatcher(
    py::array_t<double, py::array::c_style> X_obj,
    py::object tau_obj,
    py::object mask_obj,
    double dt, int k, int n_threads,
    bool full)   // true → (IF,Leak,dI); false → IF only
```

- Broadcasts scalar `tau` to array (existing logic, shared for both branches).
- If `full && mask.is_none()` → call function 1.
- If `full && mask set`      → call function 2.
- If `!full && mask.is_none()` → call function 3.
- If `!full && mask set`      → call function 4.

---

## ouxinfo.cpp changes

1. Replace `#include "information_flow.hpp"` with `#include "information_flow.cpp"`.
2. Delete the three moved functions (`information_flow_causal_map_wrapper`, `information_flow_causal_map_mask_wrapper`, `information_flow_causal_map_dispatcher`).
3. Update the existing `m.def` for `information_flow_causal_map` to add the new argument:
   ```cpp
   m.def("information_flow_causal_map", &information_flow_causal_map_dispatcher,
         py::arg("X"),
         py::arg("tau")       = py::int_(1),
         py::arg("mask")      = py::none(),
         py::arg("dt")        = 1.0,
         py::arg("k")         = 5,
         py::arg("n_threads") = 1,
         py::arg("full")      = true,   // NEW
         R"doc(...)doc");
   ```
   Update the docstring to document `full` and the two return types.

---

## Verification

```bash
./rebuild.sh
python - <<'EOF'
import numpy as np
from ouxinfo import information_flow_causal_map
rng = np.random.default_rng(0)
X = rng.standard_normal((4, 200))

# full=True (default): returns tuple
IF, Leak, dI = information_flow_causal_map(X)
assert IF.shape == (4, 4)

# full=False: returns IF only
IF_only = information_flow_causal_map(X, full=False)
assert IF_only.shape == (4, 4)
np.testing.assert_allclose(IF, IF_only, rtol=1e-10)
print("OK")
EOF

pytest tests/ -v
```
