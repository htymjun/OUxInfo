# Plan: Accelerate `mutual_information.hpp`

## Context

`mutual_info`, `mutual_info_Thei`, and `conditional_mutual_info` are KSG/KNN estimators. Each sample `i` requires one kNN search (joint space) then 2–3 radius searches (marginal spaces), followed by a sequential digamma accumulation pass. Profiling reveals four performance issues that can be fixed without changing the algorithm.

## Critical file

- **Target**: `ouxinfo/mutual_information.hpp`
- **nanoflann is not modified** — its `findNeighbors<RESULTSET>()` template already supports custom result sets.

---

## Bottlenecks identified

| # | Issue | Function(s) affected |
|---|-------|----------------------|
| A | `radiusSearch` stores results into `std::vector<ResultItem>` then a second loop counts them — heap allocation + redundant iteration per sample | `mutual_info`, `mutual_info_Thei`, `conditional_mutual_info` |
| B | A second sequential O(N) pass reads `nX[i]`/`nY[i]` arrays to compute digamma sums — needless cache miss after the parallel phase | all three |
| C | `mutual_info_Thei` has **no OpenMP** at all; per-iteration `matches_X`/`matches_Y` vectors are declared inside the loop | `mutual_info_Thei` |
| D | `radiusSearch` is called with `sorted=true` (default) — sorting O(M log M) results we only count | `mutual_info`, `mutual_info_Thei`, `conditional_mutual_info` |

---

## Changes

### 1. Add two counting result-set helpers (above the function definitions)

Replace `radiusSearch` + filter loop with a zero-allocation counting traversal via `findNeighbors`.

```cpp
// Counts points strictly inside radius, excluding self (for standard MI).
template<typename T>
struct StrictRadiusCountSet {
    using DistanceType = T;
    T radius; size_t self_idx; size_t count = 0;
    StrictRadiusCountSet(T r, size_t s) : radius(r), self_idx(s) {}
    void init() { count = 0; }
    bool full() const { return false; }
    bool addPoint(T dist, size_t idx) {
        if (idx != self_idx && dist < radius) ++count;
        return true;
    }
    T worstDist() const { return radius; }
};

// Counts points strictly inside radius, excluding the Theiler window (for MI_Thei).
template<typename T>
struct TheilerRadiusCountSet {
    using DistanceType = T;
    T radius; int center; int thei; size_t count = 0;
    TheilerRadiusCountSet(T r, int c, int w) : radius(r), center(c), thei(w) {}
    void init() { count = 0; }
    bool full() const { return false; }
    bool addPoint(T dist, size_t idx) {
        if (std::abs((int)idx - center) > thei && dist < radius) ++count;
        return true;
    }
    T worstDist() const { return radius; }
};
```

`findNeighbors(countSet, query, SearchParameters(0, false))` calls `addPoint` for every candidate nanoflann visits; `worstDist()` provides the pruning radius so nanoflann never visits nodes beyond `eps`.

### 2. Refactor `mutual_info` (lines 52–97)

Replace:
- `std::vector<int> nX(N,0), nY(N,0)` — eliminated
- `std::vector<ResultItem> matches_X, matches_Y` inside parallel block — eliminated
- `radiusSearch(..., matches_X, SearchParameters(0))` + filter loop — replaced
- Post-loop digamma accumulation pass — merged into parallel loop via reduction

```cpp
T sum_psi_nX = 0, sum_psi_nY = 0;
int valid_pts = 0;
#pragma omp parallel reduction(+:sum_psi_nX, sum_psi_nY, valid_pts)
{
    std::vector<size_t> ret_index(k+1);
    std::vector<T>      out_dist(k+1);
    #pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
        KNNResultSet<T> resultSet(k+1);
        resultSet.init(ret_index.data(), out_dist.data());
        index_XY.findNeighbors(resultSet, &XY[i*dxy], SearchParameters(0, false));
        T eps = out_dist[k];
        StrictRadiusCountSet<T> rs_X(eps, i), rs_Y(eps, i);
        index_X.findNeighbors(rs_X, &X[i*dx], SearchParameters(0, false));
        index_Y.findNeighbors(rs_Y, &Y[i*dy], SearchParameters(0, false));
        int cx = rs_X.count, cy = rs_Y.count;
        if (cx > 0 && cy > 0) {
            sum_psi_nX += digamma(cx + 1.0);
            sum_psi_nY += digamma(cy + 1.0);
            ++valid_pts;
        }
    }
}
T I = 0;
if (valid_pts > 0)
    I = digamma(k) - sum_psi_nX/valid_pts - sum_psi_nY/valid_pts + digamma(N);
```

### 3. Refactor `mutual_info_Thei` (lines 127–192)

Add OpenMP (currently none), move all thread-local state outside the inner loop, use `TheilerRadiusCountSet` for radius counting.

- `std::vector<int> nX, nY, nN` — eliminated (accumulate inline)
- `ret_index`, `out_dist` — declared once per thread (outside `for`)
- `matches_X`, `matches_Y` (declared *inside* the loop today) — eliminated entirely
- Add `#pragma omp parallel` + `#pragma omp for schedule(static)` + reductions for `sum_psi_nX`, `sum_psi_nY`, `sum_psi_nN`, `valid_pts`

```cpp
T sum_psi_nX = 0, sum_psi_nY = 0, sum_psi_nN = 0;
int valid_pts = 0;
#pragma omp parallel reduction(+:sum_psi_nX, sum_psi_nY, sum_psi_nN, valid_pts)
{
    std::vector<size_t> ret_index(num_search);
    std::vector<T>      out_dist(num_search);
    #pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
        int window_start = std::max(0, i - Thei);
        int window_end   = std::min(i + Thei + 1, N);
        int nN_i = N - (window_end - window_start);
        size_t num_results = index_XY.knnSearch(&XY[i*dxy], num_search,
                                                ret_index.data(), out_dist.data());
        T eps = 0;
        int valid_k_count = 0;
        for (size_t m = 0; m < num_results; m++) {
            if (std::abs((int)ret_index[m] - i) > Thei) {
                if (++valid_k_count == k) { eps = out_dist[m]; break; }
            }
        }
        TheilerRadiusCountSet<T> rs_X(eps, i, Thei), rs_Y(eps, i, Thei);
        index_X.findNeighbors(rs_X, &X[i*dx], SearchParameters(0, false));
        index_Y.findNeighbors(rs_Y, &Y[i*dy], SearchParameters(0, false));
        int cx = rs_X.count, cy = rs_Y.count;
        if (cx > 0 && cy > 0) {
            sum_psi_nX += digamma(cx + 1.0);
            sum_psi_nY += digamma(cy + 1.0);
            sum_psi_nN += digamma(nN_i + 1.0);
            ++valid_pts;
        }
    }
}
T I = 0;
if (valid_pts > 0)
    I = digamma(k) - sum_psi_nX/valid_pts - sum_psi_nY/valid_pts + sum_psi_nN/valid_pts;
```

### 4. Refactor `conditional_mutual_info` (lines 233–288)

Same pattern as `mutual_info`:
- Remove `nZ`, `nYZ`, `nXZ` arrays and the post-loop pass
- Remove `matches_Z/YZ/XZ` vectors from the parallel block
- Use `StrictRadiusCountSet<T>` for all three radius searches
- Add OMP reduction for `sum_psi_nZ`, `sum_psi_nXZ`, `sum_psi_nYZ`, `valid_pts`

---

## What is NOT changed

- No modifications to `nanoflann.hpp` (not necessary — `findNeighbors` is already a template)
- No change to the KSG algorithm, digamma formula, or public API
- `mutual_info_Thei` `num_search` heuristic is preserved

---

## Verification

```bash
./rebuild.sh
pytest tests/test_MI.py -v
```

All existing tests in `test_MI.py` must pass, confirming numeric results are unchanged. For a runtime check:

```bash
python tests/test_speed.py   # if it covers MI, or:
python - <<'EOF'
import numpy as np, time
from ouxinfo import mutual_information
rng = np.random.default_rng(0)
N = 10000
x = rng.standard_normal((N, 1))
y = 0.8*x + 0.6*rng.standard_normal((N, 1))
t0 = time.perf_counter()
mutual_information(x, y, k=5)
print(f"MI: {time.perf_counter()-t0:.3f}s")
EOF
```
