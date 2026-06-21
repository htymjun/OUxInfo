# Fix: Integer Overflow in Nx * Nx Expressions

**Date:** 2026-06-10

## Context

In `ouxinfo/information_flow.cpp`, `Nx` (and `N`) are declared as `int` (32-bit signed). When `Nx > ~46,340`, the expression `Nx * Nx` overflows, wrapping to a negative value. This causes incorrect `std::vector::assign` sizes, invalid pointer arithmetic in `std::fill`, and wrong array indexing — all silent undefined behaviour.

The fix is to change `Nx` and `N` to `ssize_t` (or equivalently `ptrdiff_t`) at their declaration sites so that all arithmetic derived from them uses 64-bit signed integers.

---

## Affected Locations in `ouxinfo/information_flow.cpp`

### Function `information_flow_causal_map_mask_wrapper` (lines ~274–381)

| Line | Expression | Fix |
|------|-----------|-----|
| 286 | `int Nx = static_cast<int>(...)` | → `ssize_t Nx = static_cast<ssize_t>(...)` |
| 287 | `int Nt = static_cast<int>(...)` | → `ssize_t Nt = static_cast<ssize_t>(...)` |
| 302 | `mask_default.assign(Nx * Nx, 1)` | safe once Nx is ssize_t |
| 303 | `mask_default[i * Nx + i]` | safe once Nx is ssize_t |
| 323–325 | `IF + Nx * Nx`, `dI + Nx * Nx`, `Leak + Nx * Nx` | safe once Nx is ssize_t |
| 328 | `IF[v*Nx+v]` etc. | safe once Nx is ssize_t |

Loop variables `int i`, `int v`, `int a`, `int b` used with `Nx` also need to be `ssize_t` to avoid sign-extension surprises in index arithmetic.

### Function `information_flow_only_causal_map_mask_wrapper` (lines ~462–553)

| Line | Expression | Fix |
|------|-----------|-----|
| 474 | `int Nx = static_cast<int>(...)` | → `ssize_t Nx = static_cast<ssize_t>(...)` |
| 475 | `int Nt = static_cast<int>(...)` | → `ssize_t Nt = static_cast<ssize_t>(...)` |
| 490 | `mask_default.assign(Nx * Nx, 1)` | safe once Nx is ssize_t |
| 491 | `mask_default[i * Nx + i]` | safe once Nx is ssize_t |
| 507 | `IF + Nx * Nx` | safe once Nx is ssize_t |
| 510 | `IF[v*Nx+v]` | safe once Nx is ssize_t |

### Non-mask wrappers (lines ~149, ~403)

| Line | Expression | Fix |
|------|-----------|-----|
| 149 | `int N = static_cast<int>(...)` | → `ssize_t N = static_cast<ssize_t>(...)` |
| 150 | `int Nt = ...` | → `ssize_t Nt = ...` |
| 165 | `int num_pairs = N * (N - 1) / 2` | → `ssize_t num_pairs = ...` (intermediate also overflows) |
| 403 | `int N = static_cast<int>(...)` | → `ssize_t N = static_cast<ssize_t>(...)` |
| 404 | `int Nt = ...` | → `ssize_t Nt = ...` |
| 415 | `int num_pairs = N * (N - 1) / 2` | → `ssize_t num_pairs = ...` |

Loop variables referencing `N` changed from `int` to `ssize_t` at lines ~168, ~172, ~418, ~422 as well.

---

## Implementation Steps

1. In each of the four function bodies, change:
   - `int Nx` / `int N` / `int Nt` / `int num_pairs` declarations to `ssize_t`.
   - `static_cast<int>(...)` to `static_cast<ssize_t>(...)` at those declarations.
   - Loop variables `int i`, `int v`, `int a`, `int b` that are multiplied by `Nx`/`N` to `ssize_t`.
   - Leave `int* tau_arr`, `int k`, `int n_threads`, and pybind11 shape comparisons as `int` — they are not involved in `Nx*Nx` products.

2. The `py::array_t<double> IF_map({Nx, Nx})` calls at lines 317–319 and 505 accept `ssize_t` arguments natively (pybind11 uses `ssize_t` for shape), so no change needed there.

3. No changes needed in `mutual_information.hpp`, `shannon_entropy.hpp`, or `ouxinfo.cpp` — those use `N * dxy` etc. where `dxy` is small (number of dimensions), so overflow is not practically reachable there.

---

## Verification

```bash
# Rebuild
./rebuild.sh

# Smoke test with a large Nx (e.g. Nx=50000 would overflow with int)
python - <<'EOF'
import numpy as np, ouxinfo as ou
Nx, Nt = 100, 200   # normal test first
X = np.random.randn(Nx, Nt)
tau = np.ones(Nx, dtype=np.int32)
result = ou.information_flow_causal_map_mask(X, None, tau)
print("shape:", result[0].shape)   # should be (100, 100)
EOF

# Run existing test suite
pytest tests/
```
