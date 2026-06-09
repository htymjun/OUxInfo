# Plan: Unify return signatures of `information_flow_causal_map` (masked and unmasked)

## Context

`information_flow_causal_map` is a single Python-facing function backed by a dispatcher that routes to one of two C++ wrappers:

- **No mask** (`information_flow_causal_map_wrapper`, lines 408–525): returns `py::make_tuple(IF_map, Leak_map, dI_map)` — a 3-tuple of (N×N) arrays.
- **With mask** (`information_flow_causal_map_mask_wrapper`, lines 528–596): returns a single `py::array_t<double>` (IF values only, zeros where mask is False).

The inconsistency means callers must know whether they passed a mask to decide how to unpack. This needs to be fixed: both paths should return `(IF_map, Leak_map, dI_map)`.

---

## Changes

### 1. `information_flow_causal_map_mask_wrapper` — [ouxinfo/ouxinfo.cpp:528](ouxinfo/ouxinfo.cpp#L528)

**Restructure from ordered-pair iteration to unordered-pair iteration** (matching the full wrapper) so that Leak — which requires both IF directions — can be computed consistently.

Replace the body with the following logic:

1. **Allocate three output arrays** `IF_map`, `dI_map`, `Leak_map`, all zero-initialized (shape N×N).
2. **Set diagonals to NaN** (matches full wrapper convention).
3. **Collect unordered pairs** `{a, b}` with `a < b` where `mask[a,b] || mask[b,a]`.
4. **Per pair in parallel**, compute (using `mutual_info` calls, mirroring the full wrapper math exactly):
   - `I_b  = MI(Xb[:Neff_b], Xa[:Neff_b])` — base MI, `Neff_b = Nt − tau_b` (larger-index tau)
   - `if_ab = (MI(Xb[:Neff_b], Xa[tau_b:]) − I_b) / dt` — IF from Xb→Xa = `IF[a,b]`
   - `I_a = (tau_a == tau_b) ? I_b : MI(Xa[:Neff_a], Xb[:Neff_a])`
   - `if_ba = (MI(Xa[:Neff_a], Xb[tau_a:]) − I_a) / dt` — IF from Xa→Xb = `IF[b,a]`
   - `di_val = (MI(Xb[tau_b:], Xa[tau_b:]) − I_b) / dt` — symmetric dI
   - `leak_val = di_val − if_ba − if_ab`
5. **Store into output arrays**: only fill `IF_map[a,b]` if `mask[a,b]` (else stays 0), likewise `IF_map[b,a]`; store `dI` and `Leak` symmetrically for all masked entries.
6. **Return `py::make_tuple(IF_map, Leak_map, dI_map)`** (change return type from `py::array_t<double>` to `py::object`).

This computation is mathematically identical to the full wrapper for equal-tau inputs — verified by tracing each MI call against `information_flow_causal_map_wrapper`.

### 2. Dispatcher comment — [ouxinfo/ouxinfo.cpp:623](ouxinfo/ouxinfo.cpp#L623)

Update the comment `// Mask provided: masked computation (returns single array)` → `(returns tuple of three arrays)`.

### 3. Pybind11 docstring — [ouxinfo/ouxinfo.cpp:851](ouxinfo/ouxinfo.cpp#L851)

Update the **With mask** return description from:
> `Returns : ndarray of shape (N, N).`

to:
> `Returns : tuple of three ndarray of shape (N, N). (IF_map, Leak_map, dI_map).`

### 4. Tests — [tests/test_IF_mask.py](tests/test_IF_mask.py)

Every call that does `IF = information_flow_causal_map(..., mask=..., ...)` must be updated to unpack the tuple:

| Line | Before | After |
|------|--------|-------|
| 48 | `IF_mask = information_flow_causal_map(X, tau, mask=mask, ...)` | `IF_mask, _, _ = ...` |
| 65 | `IF_mask = information_flow_causal_map(X, 1, mask=mask, ...)` | `IF_mask, _, _ = ...` |
| 82 | `IF = information_flow_causal_map(X, 1, mask=mask, ...)` | `IF, _, _ = ...` |
| 101 | `IF = information_flow_causal_map(X, 1, mask=mask, ...)` | `IF, _, _ = ...` |
| 117–118 | `IF_1 = ...` / `IF_4 = ...` | `IF_1, _, _ = ...` / `IF_4, _, _ = ...` |

---

## Verification

```bash
# Rebuild the C++ extension
./rebuild.sh

# Run all mask-related tests
pytest tests/test_IF_mask.py -v

# Run full test suite to check for regressions
pytest tests/ -v
```

The existing `test_full_mask_matches_information_flow_causal_map` test already validates that the mask path and full path agree on IF values — it will continue to do so after the change (just requiring the unpack update).
