# Plan: Speed benchmarks for ouxinfo vs infomeasure

## Goal

Add pytest speed benchmarks to `tests/` that measure and compare ouxinfo vs infomeasure wall-clock time for mutual information and transfer entropy across sample sizes, and save a log-log plot to `docs/speed_comparison.png`.

## Approach

- `tests/test_speed.py`: single test `test_speed_comparison` that benchmarks MI and TE for `N in [1000, 2000, 5000, 10000]` with a warmup call before each measurement.
- Uses `pytest.importorskip("infomeasure")` at module level — skips gracefully if infomeasure is not installed.
- Asserts ouxinfo is faster than infomeasure at N=10000 for both MI and TE.
- Saves a 2-panel log-log figure to `docs/speed_comparison.png`.

## Files

- `tests/test_speed.py` — new
- `.github/workflows/test.yml` — added `infomeasure` to `pip install`
- `docs/speed_comparison.png` — generated artifact (not committed)

## infomeasure API

| Function | Call |
|---|---|
| MI | `mutual_information(x, y, approach="ksg", k=K)` (1D arrays) |
| TE | `transfer_entropy(x, y, k=K, noise_level=0, prop_time=1, approach='ksg')` (1D arrays) |
