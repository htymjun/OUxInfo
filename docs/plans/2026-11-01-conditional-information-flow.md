# Conditional information flow

Status: implemented on branch `fix-knn-chebyshev-exactness` (2026-11-01), not yet committed.

## Goal

Condition the information flow on a user-supplied series Z, for both the single-pair
`information_flow` and `information_flow_causal_map`, via an optional argument `Z=None`.
Existing calls without `Z` are unchanged.

## Definition

    T_{X->Y|Z} = [I(X_t; Y_{t+tau} | Z_t) - I(X_t; Y_t | Z_t)] / dt,   t = 0 .. N - tau - 1

Causal map with `full=True`:

    dI[a, b] = [I(x_a,t+tau; x_b,t+tau | Z_t) - I(x_a,t; x_b,t | Z_t)] / dt
    Leak     = dI - IF - IF.T

Z is held at time t in every term. dI is therefore the change of I(X; Y | Z_t) caused by the
evolution of X and Y only, and Leak keeps its unconditional meaning. The alternative
(Z_{t+tau} in the dI term) would make dI the total derivative and move the Z-evolution
contribution into Leak.

Consequences used as exact checks:

- Z = Y: the second term is I(X; Y | Y) = 0, so T_{X->Y|Y} = TE(X -> Y) with m = 1.
- Z = (Y, W): conditional transfer entropy I(X_t; Y_{t+tau} | Y_t, W_t).
- Z = const: every Z distance is 0, n_z = N - 1, the CMI counts reduce to the MI counts and
  T_{X->Y|Z} equals the unconditional estimate.

## Implementation (`ouxinfo/information_flow.cpp`)

- `CountSpace<T>`: owns a row-major copy of concatenated blocks such as (X_i, Z_i) and answers
  strict Chebyshev radius counts (sorted array in 1-D, exact `ChebyshevKDTree` otherwise).
- `cmi_with_prebuilt(sZ, sXZ, sYZ, k)`: KSG / Frenzel–Pompe CMI; the joint XYZ points are
  assembled from sXZ and the Y part of sYZ, so joint and marginal spaces cannot disagree.
  Returns NaN when N <= k (the unconditional path reads an unset k-th distance there).
- `conditional_information_flow`: shares the Z and XZ spaces between both CMIs.
- `information_flow_causal_map_cond_wrapper`: one routine for mask / no mask, `full`, and
  2-D or 3-D X. Per-variable spaces (x_v,t, Z_t) and (x_v,t+tau, Z_t) and the Z_t space are
  built once (only for variables that appear in a requested pair); each CMI builds only its
  joint tree. Z is (Nt,) or (Nt, dz).
- The existing `conditional_mutual_info` (used by `transfer_entropy`) is left untouched, so
  the Z = Y identity compares two independent CMI implementations.
- Negative tau now raises in `information_flow` and `information_flow_causal_map` (it read
  out of bounds before).

## Verification (`tests/test_IF_conditional.py`)

1. Exhaustive-distance reference: single pair for (dx, dy, dz) in {(1,1,1), (1,1,2), (2,1,1),
   (1,3,2)}, k in {1, 4}, tau in {1, 3}; all map paths (no mask / full mask / sparse mask,
   `full`, 1 and 3 threads, dx in {1, 2}, dz in {1, 2}, 1-D Z) including Leak and dI.
   Agreement < 1e-10 (observed < 1e-14).
2. Exact identities above, < 1e-12 (observed ~1e-14), including m = 2, lag = 2 embedding.
3. Closed-form Gaussian VAR(1) values, 16 trials of N = 1e4, tolerance 0.01 nats.
4. Mutation check: six injected faults (Z shifted by tau in the lagged term or in the map,
   swapped direction, unlagged variable in dI, non-strict 1-D count, Leak sign) each make the
   suite fail.

### Bias study (k = 5, tau = 1 unless noted; bias = mean - exact, z = bias / se)

Systems (s = (X, Y, W), unit noise): common driver W -> X, W -> Y; chain X -> W -> Y
(couplings 0.9); direct X -> Y plus driver W.

| case | quantity | exact | N = 1e3 (400 trials) | N = 1e4 (100) | N = 1e5 (10) |
|---|---|---|---|---|---|
| driver | IF | -0.0543 | -0.0025 (z -1.9) | -0.0009 (-0.9) | +0.0005 (+0.6) |
| driver | IF\|W | -0.0768 | -0.0044 (-3.8) | -0.0004 (-0.5) | -0.0004 (-0.4) |
| driver | TE = IF\|Y | 0.0148 | -0.0001 (-0.2) | +0.0001 (+0.1) | +0.0006 (+1.2) |
| driver | cTE = IF\|Y,W | 0 | -0.0009 (-1.6) | -0.0003 (-0.7) | -0.0000 (-0.0) |
| chain | IF | 0.0378 | +0.0000 (0.0) | +0.0015 (+2.0) | -0.0009 (-1.1) |
| chain | IF\|W | -0.0001 | +0.0005 (+0.5) | -0.0003 (-0.4) | +0.0004 (+0.5) |
| chain | TE | 0.0389 | -0.0019 (-2.3) | +0.0009 (+1.6) | +0.0007 (+1.5) |
| chain | cTE | 0 | -0.0018 (-3.1) | -0.0004 (-1.0) | -0.0007 (-1.4) |
| chain, tau = 2 | IF | 0.2263 | +0.0024 (+1.4) | +0.0000 (0.0) | +0.0016 (+1.1) |
| chain, tau = 2 | IF\|W | 0.1641 | -0.0007 (-0.5) | +0.0017 (+1.5) | -0.0005 (-0.9) |
| chain, tau = 2 | TE | 0.2295 | -0.0014 (-1.0) | +0.0026 (+3.0) | +0.0010 (+0.9) |
| chain, tau = 2 | cTE | 0.1671 | -0.0105 (-10.0) | +0.0007 (+1.0) | +0.0012 (+1.5) |
| direct | IF | 0.1170 | -0.0030 (-2.2) | +0.0019 (+2.1) | +0.0006 (+1.2) |
| direct | IF\|W | 0.1006 | -0.0001 (-0.1) | +0.0026 (+3.3) | +0.0002 (+0.2) |
| direct | TE | 0.1224 | -0.0005 (-0.5) | +0.0022 (+3.2) | -0.0002 (-0.4) |
| direct | cTE | 0.1046 | -0.0039 (-4.4) | +0.0015 (+2.4) | +0.0012 (+2.0) |

The largest deviation (-0.0105 at N = 1e3 for the 4-D joint space of cTE) vanishes by
N = 1e4; at N = 1e5 every |bias| <= 0.0016 nats. Quantities from the same case share the
simulated series, so their z values are correlated. The unconditional estimator (validated
before) shows the same size of deviation, so the conditional path adds no visible bias
beyond the usual KSG finite-sample bias in the larger joint space.
