# Code audit of the installable package (2026-11-01)

Status: fixed on branch `fix-knn-chebyshev-exactness`, not yet committed. Follows
`2026-11-01-conditional-information-flow.md` (which also records the N <= k → NaN fix).
Method: own review of `ouxinfo/*.py` plus an independent review of the C++ core; every
crash was reproduced before and after the fix in a subprocess; tests run on the -Ofast
build and under AddressSanitizer + UndefinedBehaviorSanitizer.

## Crashes (interpreter killed)

| input | before | now |
|---|---|---|
| NaN in any input | segfault in the 1-D sorted-array path (`std::sort` with NaN), garbage on the kd-tree path (H = -0.61 instead of 2.84) | ValueError |
| ±inf in any input | self-count -1 → `digamma(0)` throws inside an OpenMP region → `std::terminate` | ValueError |
| zero columns (H, KL, MI, CMI, TE, IF, 3-D maps) | segfault (dim-0 kd-tree) | ValueError |
| negative tau / m < 1 / lag < 1 (TE, TE map), k < 1, Thei < 0 | out-of-bounds reads | ValueError |
| KL with X and Y of different dimension | out-of-bounds reads | RuntimeError |
| empty causal map (N = 0) | read of tau[0] | ValueError |

The finite check tests the exponent bits, because -ffinite-math-only lets the compiler fold
`std::isfinite()` to true.

## Silently wrong results

- `auto_mutual_info` returned the array index of the first MI minimum instead of the lag
  (one too small: 8 instead of 9 for a period-40 sine).
- `backward_transfer_entropy` ignored `trial` and always subtracted one random surrogate, so
  in `TEIFL` the map `mTE - bTE` compared an uncorrected with a surrogate-corrected TE.
- `TEIFL(multi=True)`: `mmap` lagged the embedding actually used for `mTE`/`bTE` by one and
  stayed 0 when convergence happened at m = 2; now it is the m used for both.
- `KL_div` clipped the estimate at 0: upward bias (47 % of identical-distribution trials gave
  exactly 0, mean +0.017) and NaN (duplicates) became 0. Now unclipped.
- Duplicates: MI/CMI skipped eps = 0 points but returned 0.0 if all were skipped; H returned
  -inf; KL 0. Now all estimators skip zero-distance points and return NaN if none is left.
  Ties bias every KSG estimate even without eps = 0 (bivariate Gaussian r = 0.8 rounded to
  0.01: MI 0.674 vs 0.511; H(x) 3.44 vs 1.42), so docstrings now recommend low-amplitude
  noise for quantized data (Kraskov et al. 2004).
- Theiler MI with a huge window: `k + 2 Thei + 1` overflowed and returned 0.0; now NaN.
- Causal maps called `omp_set_num_threads(n_threads)` and never restored it, so every later
  estimator call ran with that thread count; now restored (RAII guard), n_threads < 1 raises.
- TE map surrogates were seeded per thread under `schedule(dynamic)`, so results depended on
  n_threads; now seeded per pair (`seed`, j, i). `transfer_entropy` gained `seed` (default -1 =
  random, as before).

## Other

- 32-bit index arithmetic (`i*Nt*dx`, `i*d`, `N*dxy`, `j*N+i`) wrapped for > 2^31 elements
  (e.g. 1100 grid points × 2e6 steps); now 64-bit.
- nanoflann 1.7.1 builds the index in the constructor; the explicit `buildIndex()` calls
  built every tree twice. Removed (results unchanged, brute-force tests pass).
- `shannon_entropy` summed with an atomic; now a reduction like the other estimators.
  All estimators still vary in the last bit (~1e-16) between runs because OpenMP combines
  per-thread partial sums in arbitrary order.
- `entropy_rate(tau=0)` sliced `x[:-0]` (empty); `TEIFL(max_m < 2)` raised NameError
  (`InvalidInputError` undefined). Both raise ValueError now.
- Docstrings corrected: TE does not embed the source; IF is the change of I(X;Y) caused by Y's
  evolution (not "due to X"); a mask entry selects the pair and both directions are filled;
  tau must be equal for all variables.

## Open questions (not changed)

- Every rate is divided by `dt`, not `tau * dt`; with tau > 1 the result is per lag, not per
  unit time, unless the caller passes dt = tau Δt.
- Tied data: skipping eps = 0 points keeps mostly-continuous data usable but returns a
  plausible-looking number for heavily quantized data; the alternative is NaN as soon as any
  point has eps = 0.

Tests: tests/test_api_consistency.py, test_knn_exact.py (NaN / argument checks).
