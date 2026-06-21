# Plan: Apply prebuilt-sorted-array optimization to full=True wrappers

## Context

The `full=False` mask wrapper (`information_flow_only_causal_map_mask_wrapper`) was already optimized:
precomputed per-variable `SortedArray1D` + `mi_with_prebuilt_x` + simplified to 3 MI calls/pair.

The user wants the same treatment applied to `full=True`, which routes through two wrappers in
`ouxinfo/information_flow.cpp`:
- `information_flow_causal_map_mask_wrapper` (lines ~235–343) — mask path
- `information_flow_causal_map_wrapper` (lines ~130–228) — no-mask path

Both use `mutual_info()` which rebuilds all marginal structures on every call. Under the same
invariants (tau_i == tau_j, symmetric mask), the per-pair call pattern for the mask wrapper is:

| call | structures built |
|---|---|
| `mutual_info(xb, xa)`         | XY tree + Xb sorted + Xa sorted |
| `mutual_info(xb, xa+tau)`     | XY tree + **Xb again** + Xa[τ:] sorted |
| `I_a = I_b` (ternary skipped) | — |
| `mutual_info(xa, xb+tau)`     | XY tree + **Xa again** + Xb[τ:] sorted |
| `mutual_info(xb+tau, xa+tau)` | XY tree + **Xb[τ:] again** + **Xa[τ:] again** |

**15 builds per pair, 6 redundant.** With prebuilt sorted arrays: 8 builds per pair (4 XY trees + 4 Y
marginals), eliminating all X-role redundancy.

## Two precomputed sorted arrays per variable

```cpp
// sa_vars[v]     covers X[v][0   : Neff]        (unlagged, used in I_ab, Ilag_ab, Ilag_ba)
// sa_lag_vars[v] covers X[v][tau : tau + Neff]   (lagged,   used in Ilag_dI)

int tau = tau_arr[0];   // tau_i == tau_j always
int Neff = Nt - tau;
std::vector<SortedArray1D<double>> sa_vars(Nx), sa_lag_vars(Nx);
if (Neff > 0) {
  for (int v = 0; v < Nx; ++v) {
    sa_vars[v].build(X + v * Nt, Neff);
    sa_lag_vars[v].build(X + v * Nt + tau, Neff);
  }
}
my_kd_tree_t<double>* const no_kd = nullptr;
```

Both arrays are read-only during the parallel loop → thread-safe.

## Simplified inner loop (mask wrapper)

Replaces lines ~297–340. Removes:
- `tau_a`, `tau_b`, `Neff_a`, `Neff_b` locals (collapsed to single `tau`/`Neff`)
- `(tau_a == tau_b) ? I_b : ...` ternary (always true → always I_ab)
- `if (mask[a*Nx+b])` / `if (mask[b*Nx+a])` conditional writes (symmetric mask assumed)
- `if (Neff_a <= 0 || Neff_b <= 0)` per-direction NaN assignments (single assignment)

```cpp
for (int p = 0; p < (int)pairs.size(); ++p) {
  int a = pairs[p].first;
  int b = pairs[p].second;
  double* xa = X + a * Nt;
  double* xb = X + b * Nt;

  if (Neff <= 0) {
    IF[a*Nx+b] = IF[b*Nx+a] = dI[a*Nx+b] = dI[b*Nx+a] =
      Leak[a*Nx+b] = Leak[b*Nx+a] = std::numeric_limits<double>::quiet_NaN();
    continue;
  }

  double I_ab    = mi_with_prebuilt_x(xb, xa,       k, 1, 1, Neff, &sa_vars[b],     no_kd);
  double Ilag_ab = mi_with_prebuilt_x(xb, xa + tau, k, 1, 1, Neff, &sa_vars[b],     no_kd);
  double Ilag_ba = mi_with_prebuilt_x(xa, xb + tau, k, 1, 1, Neff, &sa_vars[a],     no_kd);
  double Ilag_dI = mi_with_prebuilt_x(xb + tau, xa + tau, k, 1, 1, Neff, &sa_lag_vars[b], no_kd);

  double if_ab    = (Ilag_ab - I_ab) / dt;
  double if_ba    = (Ilag_ba - I_ab) / dt;
  double di_val   = (Ilag_dI - I_ab) / dt;
  double leak_val = di_val - if_ab - if_ba;

  IF[a*Nx+b]   = if_ab;
  IF[b*Nx+a]   = if_ba;
  dI[a*Nx+b]   = dI[b*Nx+a]   = di_val;
  Leak[a*Nx+b] = Leak[b*Nx+a] = leak_val;
}
```

**Note:** The original code had a second `#pragma omp parallel for` loop to compute Leak in the mask wrapper — wait, actually looking at the code, the mask wrapper already computes Leak inline (no second loop). The no-mask wrapper does have a second loop (lines 219–226 of `information_flow_causal_map_wrapper`).

## Simplified inner loop (no-mask wrapper)

For `information_flow_causal_map_wrapper` (dx can be 1 or >1). Since dx>1 requires kd-tree
precomputation (more complex), keep the existing `mutual_info` path for dx>1 in a separate branch.
The dx==1 path gets the same optimization.

Pairs variable: (j, i) with j < i (existing convention preserved):

```cpp
if (dx == 1) {
  // Precompute sorted arrays (outside parallel loop)
  int tau = tau_arr[0];  // tau_i == tau_j always
  int Neff = Nt - tau;
  std::vector<SortedArray1D<double>> sa_vars(N), sa_lag_vars(N);
  if (Neff > 0) {
    for (int v = 0; v < N; ++v) {
      sa_vars[v].build(X + v * Nt, Neff);
      sa_lag_vars[v].build(X + v * Nt + tau, Neff);
    }
  }
  my_kd_tree_t<double>* const no_kd = nullptr;

  #pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < num_pairs; ++p) {
    int j = pairs[p].first;
    int i = pairs[p].second;
    double* xj = X + j * Nt;
    double* xi = X + i * Nt;

    if (Neff <= 0) {
      IF[j*N+i] = IF[i*N+j] = dI[j*N+i] = dI[i*N+j] =
        Leak[j*N+i] = Leak[i*N+j] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    double I_ij    = mi_with_prebuilt_x(xi, xj,       k, 1, 1, Neff, &sa_vars[i], no_kd);
    double Ilag_ji = mi_with_prebuilt_x(xi, xj + tau, k, 1, 1, Neff, &sa_vars[i], no_kd);
    double Ilag_ij = mi_with_prebuilt_x(xj, xi + tau, k, 1, 1, Neff, &sa_vars[j], no_kd);
    double Ilag_dI = mi_with_prebuilt_x(xi + tau, xj + tau, k, 1, 1, Neff, &sa_lag_vars[i], no_kd);

    double if_ji    = (Ilag_ji - I_ij) / dt;   // source i → target j
    double if_ij    = (Ilag_ij - I_ij) / dt;   // source j → target i
    double di_val   = (Ilag_dI - I_ij) / dt;
    double leak_val = di_val - if_ji - if_ij;

    IF[j*N+i] = if_ji;
    IF[i*N+j] = if_ij;
    dI[j*N+i] = dI[i*N+j] = di_val;
    Leak[j*N+i] = Leak[i*N+j] = leak_val;
  }
  // Second Leak loop eliminated — computed inline above
} else {
  // Existing mutual_info code for dx > 1 (unchanged)
  ...existing parallel loop...
  // Second Leak loop
  ...existing Leak loop...
}
```

Eliminating the second `#pragma omp parallel for` loop for Leak saves one full parallel loop
dispatch over all pairs (minor but clean).

## Files to modify

`ouxinfo/information_flow.cpp` only:
- `information_flow_causal_map_mask_wrapper` (lines ~297–340): replace inner loop
- `information_flow_causal_map_wrapper` (lines ~178–226): add dx==1 branch with prebuilt arrays, fold Leak into first loop

## Verification

```bash
rm -rf build/
CC=gcc CXX=g++ python setup.py build_ext --inplace
pytest tests/test_IF_mask.py -v
```

The existing tests `test_full_mask_matches_information_flow_causal_map`, `test_masked_pairs_are_zero`,
`test_causal_direction`, `test_vs_analytical`, and `test_n_threads_same_result` all exercise the
`full=True` path and will catch any regression.
