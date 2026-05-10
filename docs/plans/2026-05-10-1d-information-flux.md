# 1D Information Flux Analyzer

## Context

Implements directed information transport on a 1D spatial grid via the KSG `information_flow` estimator. For each neighboring cell interface i+½, the framework computes bidirectional information flows, net/symmetric fluxes, and estimator leak as a reliability indicator.

## Files

| File | Role |
|---|---|
| `ouxinfo/flux1d.py` | `information_flux_1d()` — main API |
| `ouxinfo/__init__.py` | exports `information_flux_1d` |
| `tests/test_flux1d.py` | unit tests |

## API

```python
result = information_flux_1d(data, dt=1.0, tau=1, k=5, n_threads=1)
```

**Input:** `data.shape = (nx, nt)`, dtype float64

**Output:** dict of `ndarray` shape `(nx-1,)`:

| Key | Definition |
|---|---|
| `J_fwd` | J_{i → i+1} |
| `J_bwd` | J_{i+1 → i} |
| `J_net` | J_fwd − J_bwd |
| `J_sym` | ½(J_fwd + J_bwd) |
| `Leak_fwd` | leak (forward direction) |
| `Leak_bwd` | leak (backward direction; equals Leak_fwd — symmetric) |

## Implementation Notes

- Uses `information_flow_causal_map(pair, taus, dt, k)` on each 2-cell pair to obtain IF + Leak in one call.
- Indexing: `IF[j, i]` = flow from variable i to j (confirmed from `ouxinfo.cpp:484`).
- Leak is symmetric per pair: `Leak[1,0] == Leak[0,1]`.
- `tau=1` is fixed per specification; exposed as a parameter for completeness.
