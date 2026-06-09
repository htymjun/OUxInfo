# Plan: Validate `ouxinfo/distributed.py` — correctness + speed

## Context

`distributed.py` provides MPI-distributed versions of `transfer_entropy_causal_map` and
`information_flow_causal_map` for HPC workloads. Before trusting these in production, we
need a test file that (1) verifies they produce the **same results** as the serial/OpenMP
C++ versions and (2) **measures and plots their relative speed**.

The C++ `information_flow_causal_map_wrapper` (lines 480–528 of `ouxinfo/ouxinfo.cpp`)
uses **identical formulas** to `information_flow_causal_map_mpi` in `distributed.py` —
both call `_core.mutual_info` with the same sliced arrays. Results should therefore be
bit-for-bit equal (no randomness when `trial=0`).

---

## New file: `tests/test_distributed.py`

### Structure

```
pytest.importorskip("mpi4py")          ← skip whole module if MPI not installed

_make_ring_data(N, Nt, coupling, seed) ← ring-coupled logistic map (adapted from tutorial)

test_te_causal_map_mpi_matches_serial  ← correctness: TE MPI vs OpenMP n_threads=1
test_if_causal_map_mpi_matches_serial  ← correctness: IF/Leak/dI MPI vs OpenMP n_threads=1
test_speed_distributed                 ← timing sweep + saves figure
```

### Correctness tests

- Data: `_make_ring_data(N=6, Nt=800, coupling=0.15, seed=42)` — ring-coupled logistic map
- `tau = np.ones(N, dtype=np.int32)` (uniform delay → exercises the `tau_i == tau_j`
  reuse-branch in both C++ and Python)
- Compare with `np.testing.assert_allclose(rtol=1e-10)` on off-diagonal entries
  (diagonal is `NaN` in both; skip via `mask = ~np.isnan(...)`)
- TE test: `transfer_entropy_causal_map_mpi` vs `transfer_entropy_causal_map(n_threads=1)`
- IF test: `information_flow_causal_map_mpi` vs `information_flow_causal_map(n_threads=1)`,
  checking all three outputs (`IF`, `Leak`, `dI`)

### Speed test

Sweep `N ∈ [4, 6, 8, 10]`, `Nt=500`, `k=5`.  For each N measure wall-clock time of:

| Label | Call |
|---|---|
| MPI (1 rank) | `transfer_entropy_causal_map_mpi(X, tau, k=K)` |
| OpenMP 1-thread | `transfer_entropy_causal_map(X, tau, k=K, n_threads=1)` |
| OpenMP N-threads | `transfer_entropy_causal_map(X, tau, k=K, n_threads=cpu_count())` |

Repeat for IF (`information_flow_causal_map_mpi` vs `information_flow_causal_map`).

Timing pattern (matches `test_speed.py`): warmup call → `time.perf_counter()` around a
single timed call.

Figure: `2×1` subplot (TE timing | IF timing), loglog, three lines each.
Save to `docs/speed_distributed.png` (matches existing `docs/speed_comparison_omp.png`).
Style: Times New Roman, STIX math, inward ticks, `font.size=16` (same as `test_speed.py`).

No assertion on speed results — the test always passes; the figure is the deliverable.

---

## Files to modify

| File | Change |
|---|---|
| `tests/test_distributed.py` | **Create** (new file) |
| `tests/conftest.py` | No change needed — speed figure saved inside the test function |

---

## Key reuse / imports

- `from ouxinfo import transfer_entropy_causal_map, information_flow_causal_map`
  (re-exported by `ouxinfo/__init__.py` from `_core`)
- `from ouxinfo.distributed import transfer_entropy_causal_map_mpi, information_flow_causal_map_mpi`
- Ring-data generator adapted from `tutorials/hpc_causal_map.py` `make_ring_data()`

---

## Verification

```bash
pytest tests/test_distributed.py -v
# Expected:
#   test_te_causal_map_mpi_matches_serial  PASSED
#   test_if_causal_map_mpi_matches_serial  PASSED
#   test_speed_distributed                 PASSED
#   docs/speed_distributed.png             saved

# If mpi4py is not installed, all three tests are SKIPPED (clean CI behavior)
```
