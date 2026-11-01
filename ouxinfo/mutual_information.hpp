#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream> // for std::cerr, std::cout
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdint>
#include <omp.h>


using namespace nanoflann;
using boost::math::digamma;

template <typename T>
using my_kd_tree_t = ChebyshevKDTree<T, PointCloud, size_t>;


// k-NN estimators need k >= 1.  With fewer than k + 1 samples there is no k-th
// neighbour (the result buffer is never filled) and the estimators return NaN.
inline void check_knn_k(int k) {
  if (k < 1) throw std::invalid_argument("k must be >= 1");
}

// NaN breaks the strict weak ordering std::sort needs (out-of-bounds reads) and inf makes
// |x_i - x_i| NaN, so non-finite input is rejected.  The exponent-bit test is used because
// -Ofast (-ffinite-math-only) lets the compiler fold std::isfinite() to true.
inline void check_finite(const double* p, size_t n, const char* name) {
  for (size_t i = 0; i < n; i++) {
    uint64_t bits;
    std::memcpy(&bits, p + i, sizeof bits);
    if ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL)
      throw std::invalid_argument(std::string(name) + " contains NaN or inf");
  }
}

// A zero-dimensional point set has no kd-tree (nanoflann indexes an empty bounding box).
inline void check_dim(long long d, const char* name) {
  if (d < 1) throw std::invalid_argument(std::string(name) + " must have at least one column");
}

// Sets the OpenMP thread count for one call and restores the previous value afterwards,
// so that n_threads of a causal map does not leak into later calls.
struct OmpThreadGuard {
  int prev;
  explicit OmpThreadGuard(int n) : prev(omp_get_max_threads()) {
    if (n < 1) throw std::invalid_argument("n_threads must be >= 1");
    omp_set_num_threads(n);
  }
  ~OmpThreadGuard() { omp_set_num_threads(prev); }
  OmpThreadGuard(const OmpThreadGuard&) = delete;
  OmpThreadGuard& operator=(const OmpThreadGuard&) = delete;
};


// ============================================================
// Zero-allocation counting result set for nanoflann::findNeighbors
// Used for >1D marginals (1D marginals use SortedArray1D below)
// ============================================================
template<typename T>
struct StrictRadiusCountSet {
  using DistanceType = T;
  T      radius;
  size_t self_idx;
  size_t count = 0;
  StrictRadiusCountSet(T r, size_t s) : radius(r), self_idx(s) {}
  void   init()            { count = 0; }
  bool   full()      const { return false; }
  bool   addPoint(T dist, size_t idx) {
    if (idx != self_idx && dist < radius) ++count;
    return true;
  }
  T    worstDist() const { return radius; }
  void sort()            {}  // required by nanoflann ResultSet interface
};


// ============================================================
// Sorted array for O(log N) 1D Chebyshev radius counting
//
// For 1D data the Chebyshev distance is |x_j - x_i|, which reduces
// to a range query on a sorted array.  This is ~5-10x faster than
// kd-tree traversal because it accesses memory sequentially.
// ============================================================
template<typename T>
struct SortedArray1D {
  std::vector<T>   vals;   // vals[j] = sorted value at position j
  std::vector<int> rank_;  // rank_[i] = position of original index i in vals

  void build(const T* data, int N) {
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return data[a] < data[b]; });
    vals.resize(N);
    rank_.resize(N);
    for (int j = 0; j < N; j++) {
      vals[j]         = data[order[j]];
      rank_[order[j]] = j;
    }
  }

  // Index range [lo, hi) of the sorted values with |vals[j] - v| < eps.
  // The binary search uses v -/+ eps, which is rounded; the edges are then re-tested
  // in difference form so that the boundary point at distance exactly eps (the k-th
  // neighbour) is excluded, as in the KSG definition.
  void strictRange(T v, T eps, int& lo, int& hi) const {
    const int n = (int)vals.size();
    lo = (int)(std::upper_bound(vals.begin(), vals.end(), v - eps) - vals.begin());
    hi = (int)(std::lower_bound(vals.begin(), vals.end(), v + eps) - vals.begin());
    while (lo > 0 && std::abs(vals[lo-1] - v) < eps) --lo;
    while (lo < n && !(std::abs(vals[lo] - v) < eps)) ++lo;
    while (hi < n && std::abs(vals[hi] - v) < eps) ++hi;
    while (hi > lo && !(std::abs(vals[hi-1] - v) < eps)) --hi;
  }

  // Count points strictly within radius of v, excluding self (original index = self).
  int countStrict(T v, T eps, int self) const {
    int lo, hi;
    strictRange(v, eps, lo, hi);
    return hi - lo - 1;  // -1 excludes self
  }

  // Count with Theiler-window exclusion.
  // Subtract points j with |j - self| <= thei that happen to fall within the radius.
  int countTheiler(T v, T eps, int self, int thei, int N) const {
    int lo, hi;
    strictRange(v, eps, lo, hi);
    int cnt = hi - lo - 1;  // -1 excludes self
    // Scan the Theiler window and remove those inside the radius range
    int win_lo = std::max(0,   self - thei);
    int win_hi = std::min(N-1, self + thei);
    for (int j = win_lo; j <= win_hi; j++) {
      if (j == self) continue;
      int r = rank_[j];
      if (r >= lo && r < hi) --cnt;
    }
    return cnt;
  }
};


// ============================================================
// mutual information
// ============================================================
template<typename T>
T mutual_info(T **X_ptr, T **Y_ptr, int k, int dx, int dy, int N) {
  if (N <= k) return std::numeric_limits<T>::quiet_NaN();
  T *X = *X_ptr;
  T *Y = *Y_ptr;
  int dxy = dx + dy;

  // --- Joint XY data ---
  std::vector<T> XY((size_t)N * dxy);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XY[(size_t)i*dxy+j]    = X[(size_t)i*dx+j];
    for (int j = 0; j < dy; j++) XY[(size_t)i*dxy+dx+j] = Y[(size_t)i*dy+j];
  }

  // --- Joint XY kd-tree (always needed) ---
  PointCloud cloud_XY;
  cloud_XY.N = N; cloud_XY.dim = dxy; cloud_XY.pts = XY.data();
  my_kd_tree_t<T> index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));

  // --- X marginal: sorted array for 1D, kd-tree for >1D ---
  SortedArray1D<T> sa_X;
  std::unique_ptr<my_kd_tree_t<T>> idx_X;
  PointCloud cloud_X;
  if (dx == 1) {
    sa_X.build(X, N);
  } else {
    cloud_X.N = N; cloud_X.dim = dx; cloud_X.pts = X;
    idx_X = std::make_unique<my_kd_tree_t<T>>(dx, cloud_X, KDTreeSingleIndexAdaptorParams(10));
  }

  // --- Y marginal: sorted array for 1D, kd-tree for >1D ---
  SortedArray1D<T> sa_Y;
  std::unique_ptr<my_kd_tree_t<T>> idx_Y;
  PointCloud cloud_Y;
  if (dy == 1) {
    sa_Y.build(Y, N);
  } else {
    cloud_Y.N = N; cloud_Y.dim = dy; cloud_Y.pts = Y;
    idx_Y = std::make_unique<my_kd_tree_t<T>>(dy, cloud_Y, KDTreeSingleIndexAdaptorParams(10));
  }

  T   sum_psi_nX = 0, sum_psi_nY = 0;
  int valid_pts  = 0;
  #pragma omp parallel reduction(+:sum_psi_nX, sum_psi_nY, valid_pts)
  {
  std::vector<size_t> ret_index(k+1);
  std::vector<T>      out_dist(k+1);
  #pragma omp for schedule(static)
  for (int i = 0; i < N; i++) {
    // --- XY space kNN (Chebyshev) ---
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XY.findNeighbors(resultSet, &XY[(size_t)i*dxy], SearchParameters(0, false));
    T eps = out_dist[k];

    // --- X marginal radius count ---
    int cx;
    if (dx == 1) {
      cx = sa_X.countStrict(X[i], eps, i);
    } else {
      StrictRadiusCountSet<T> rs(eps, (size_t)i);
      idx_X->findNeighbors(rs, &X[(size_t)i*dx], SearchParameters(0, false));
      cx = (int)rs.count;
    }

    // --- Y marginal radius count ---
    int cy;
    if (dy == 1) {
      cy = sa_Y.countStrict(Y[i], eps, i);
    } else {
      StrictRadiusCountSet<T> rs(eps, (size_t)i);
      idx_Y->findNeighbors(rs, &Y[(size_t)i*dy], SearchParameters(0, false));
      cy = (int)rs.count;
    }

    // Every point contributes, including n_x = 0 or n_y = 0 (legitimate for k = 1);
    // only degenerate points with eps == 0 (duplicates) are skipped.
    if (eps > 0) {
      sum_psi_nX += digamma(cx + 1.0);
      sum_psi_nY += digamma(cy + 1.0);
      ++valid_pts;
    }
  }
  }
  T I = std::numeric_limits<T>::quiet_NaN();  // stays NaN if every point had eps == 0
  if (valid_pts > 0)
    I = digamma(k) - sum_psi_nX/valid_pts - sum_psi_nY/valid_pts + digamma(N);
  return I;
}


// ============================================================
// mutual information with Theiler window
// ============================================================
template<typename T>
T mutual_info_Thei(T **X_ptr, T **Y_ptr, int k, int dx, int dy, int N, int Thei) {
  // The point in the middle of the series has the fewest samples outside its window,
  // N - (2*Thei + 1); below k it has no k-th neighbour.
  if ((long long)N < (long long)k + 2LL * Thei + 1) return std::numeric_limits<T>::quiet_NaN();
  T *X = *X_ptr;
  T *Y = *Y_ptr;
  int dxy = dx + dy;

  // --- Joint XY data ---
  std::vector<T> XY((size_t)N * dxy);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XY[(size_t)i*dxy+j]    = X[(size_t)i*dx+j];
    for (int j = 0; j < dy; j++) XY[(size_t)i*dxy+dx+j] = Y[(size_t)i*dy+j];
  }

  // --- Joint XY kd-tree ---
  PointCloud cloud_XY;
  cloud_XY.N = N; cloud_XY.dim = dxy; cloud_XY.pts = XY.data();
  my_kd_tree_t<T> index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));

  // --- X marginal ---
  SortedArray1D<T> sa_X;
  std::unique_ptr<my_kd_tree_t<T>> idx_X;
  PointCloud cloud_X;
  if (dx == 1) {
    sa_X.build(X, N);
  } else {
    cloud_X.N = N; cloud_X.dim = dx; cloud_X.pts = X;
    idx_X = std::make_unique<my_kd_tree_t<T>>(dx, cloud_X, KDTreeSingleIndexAdaptorParams(10));
  }

  // --- Y marginal ---
  SortedArray1D<T> sa_Y;
  std::unique_ptr<my_kd_tree_t<T>> idx_Y;
  PointCloud cloud_Y;
  if (dy == 1) {
    sa_Y.build(Y, N);
  } else {
    cloud_Y.N = N; cloud_Y.dim = dy; cloud_Y.pts = Y;
    idx_Y = std::make_unique<my_kd_tree_t<T>>(dy, cloud_Y, KDTreeSingleIndexAdaptorParams(10));
  }

  size_t num_search = (size_t)(k + 2 * Thei + 1);  // <= N, so no overflow after the check above

  T   sum_psi_nX = 0, sum_psi_nY = 0, sum_psi_nN = 0;
  int valid_pts  = 0;
  #pragma omp parallel reduction(+:sum_psi_nX, sum_psi_nY, sum_psi_nN, valid_pts)
  {
  std::vector<size_t> ret_index(num_search);
  std::vector<T>      out_dist(num_search);
  #pragma omp for schedule(static)
  for (int i = 0; i < N; i++) {
    // --- valid sample count (excluding Theiler window) ---
    int window_start = std::max(0, i - Thei);
    int window_end   = std::min(i + Thei + 1, N);
    int nN_i = N - (window_end - window_start);

    // --- find k-th XY neighbor outside the Theiler window ---
    size_t num_results = index_XY.knnSearch(&XY[(size_t)i*dxy], num_search,
                                            ret_index.data(), out_dist.data());
    T eps = T(0);
    int valid_k_count = 0;
    for (size_t m = 0; m < num_results; m++) {
      if (std::abs((int)ret_index[m] - i) > Thei) {
        if (++valid_k_count == k) { eps = out_dist[m]; break; }
      }
    }

    // --- X marginal radius count ---
    int cx;
    if (dx == 1) {
      cx = sa_X.countTheiler(X[i], eps, i, Thei, N);
    } else {
      std::vector<nanoflann::ResultItem<size_t,T>> matches;
      idx_X->radiusSearch(&X[(size_t)i*dx], eps, matches, nanoflann::SearchParameters(0, false));
      cx = 0;
      for (const auto& m : matches)
        if (std::abs((int)m.first - i) > Thei && m.second < eps) ++cx;
    }

    // --- Y marginal radius count ---
    int cy;
    if (dy == 1) {
      cy = sa_Y.countTheiler(Y[i], eps, i, Thei, N);
    } else {
      std::vector<nanoflann::ResultItem<size_t,T>> matches;
      idx_Y->radiusSearch(&Y[(size_t)i*dy], eps, matches, nanoflann::SearchParameters(0, false));
      cy = 0;
      for (const auto& m : matches)
        if (std::abs((int)m.first - i) > Thei && m.second < eps) ++cy;
    }

    // Every point contributes, including n_x = 0 or n_y = 0 (legitimate for k = 1);
    // only degenerate points with eps == 0 (duplicates) are skipped.
    if (eps > 0) {
      sum_psi_nX += digamma(cx + 1.0);
      sum_psi_nY += digamma(cy + 1.0);
      sum_psi_nN += digamma(nN_i + 1.0);
      ++valid_pts;
    }
  }
  }
  T I = std::numeric_limits<T>::quiet_NaN();  // stays NaN if every point had eps == 0
  if (valid_pts > 0)
    I = digamma(k) - sum_psi_nX/valid_pts - sum_psi_nY/valid_pts + sum_psi_nN/valid_pts;
  return I;
}


// ============================================================
// conditional mutual information I(X;Y|Z)
// ============================================================
template<typename T>
T conditional_mutual_info(T **x_ptr, T **y_ptr, T **z_ptr, int k, int dx, int dy, int dz, int N) {
  if (N <= k) return std::numeric_limits<T>::quiet_NaN();
  T *X = *x_ptr;
  T *Y = *y_ptr;
  T *Z = *z_ptr;
  int dxyz = dx + dy + dz; int dyz = dy + dz; int dxz = dx + dz;

  // --- Joint data XYZ, YZ, XZ ---
  std::vector<T> XYZ((size_t)N * dxyz), YZ((size_t)N * dyz), XZ((size_t)N * dxz);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XYZ[(size_t)i*dxyz+j]       = X[(size_t)i*dx+j];
    for (int j = 0; j < dy; j++) XYZ[(size_t)i*dxyz+dx+j]    = Y[(size_t)i*dy+j];
    for (int j = 0; j < dz; j++) XYZ[(size_t)i*dxyz+dx+dy+j] = Z[(size_t)i*dz+j];
    for (int j = 0; j < dy; j++) YZ[(size_t)i*dyz+j]          = Y[(size_t)i*dy+j];
    for (int j = 0; j < dz; j++) YZ[(size_t)i*dyz+dy+j]       = Z[(size_t)i*dz+j];
    for (int j = 0; j < dx; j++) XZ[(size_t)i*dxz+j]          = X[(size_t)i*dx+j];
    for (int j = 0; j < dz; j++) XZ[(size_t)i*dxz+dx+j]       = Z[(size_t)i*dz+j];
  }

  // --- XYZ kd-tree (always needed) ---
  PointCloud cloud_XYZ;
  cloud_XYZ.N = N; cloud_XYZ.dim = dxyz; cloud_XYZ.pts = XYZ.data();
  my_kd_tree_t<T> index_XYZ(dxyz, cloud_XYZ, KDTreeSingleIndexAdaptorParams(10));

  // --- Z marginal: sorted array for 1D, kd-tree for >1D ---
  SortedArray1D<T> sa_Z;
  std::unique_ptr<my_kd_tree_t<T>> idx_Z;
  PointCloud cloud_Z;
  if (dz == 1) {
    sa_Z.build(Z, N);
  } else {
    cloud_Z.N = N; cloud_Z.dim = dz; cloud_Z.pts = Z;
    idx_Z = std::make_unique<my_kd_tree_t<T>>(dz, cloud_Z, KDTreeSingleIndexAdaptorParams(10));
  }

  // --- YZ and XZ always use kd-trees (joint dimension > 1D in typical use) ---
  PointCloud cloud_YZ, cloud_XZ;
  cloud_YZ.N = N; cloud_YZ.dim = dyz; cloud_YZ.pts = YZ.data();
  cloud_XZ.N = N; cloud_XZ.dim = dxz; cloud_XZ.pts = XZ.data();
  my_kd_tree_t<T> index_YZ(dyz, cloud_YZ, KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t<T> index_XZ(dxz, cloud_XZ, KDTreeSingleIndexAdaptorParams(10));

  T   sum_psi_nZ = 0, sum_psi_nXZ = 0, sum_psi_nYZ = 0;
  int valid_pts  = 0;
  #pragma omp parallel reduction(+:sum_psi_nZ, sum_psi_nXZ, sum_psi_nYZ, valid_pts)
  {
  std::vector<size_t> ret_index(k+1);
  std::vector<T>      out_dist(k+1);
  #pragma omp for schedule(static)
  for (int i = 0; i < N; i++) {
    // --- XYZ space kNN (Chebyshev) ---
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XYZ.findNeighbors(resultSet, &XYZ[(size_t)i*dxyz], SearchParameters(0, false));
    T eps = out_dist[k];

    // --- Z marginal radius count ---
    int cz;
    if (dz == 1) {
      cz = sa_Z.countStrict(Z[i], eps, i);
    } else {
      StrictRadiusCountSet<T> rs(eps, (size_t)i);
      idx_Z->findNeighbors(rs, &Z[(size_t)i*dz], SearchParameters(0, false));
      cz = (int)rs.count;
    }

    // --- YZ and XZ radius counts ---
    StrictRadiusCountSet<T> rs_YZ(eps, (size_t)i), rs_XZ(eps, (size_t)i);
    index_YZ.findNeighbors(rs_YZ, &YZ[(size_t)i*dyz], SearchParameters(0, false));
    index_XZ.findNeighbors(rs_XZ, &XZ[(size_t)i*dxz], SearchParameters(0, false));
    int cyz = (int)rs_YZ.count, cxz = (int)rs_XZ.count;

    if (eps > 0) {
      sum_psi_nZ  += digamma(cz  + 1.0);
      sum_psi_nYZ += digamma(cyz + 1.0);
      sum_psi_nXZ += digamma(cxz + 1.0);
      ++valid_pts;
    }
  }
  }
  T I = std::numeric_limits<T>::quiet_NaN();  // stays NaN if every point had eps == 0
  if (valid_pts > 0)
    I = digamma(k) + sum_psi_nZ/valid_pts - sum_psi_nXZ/valid_pts - sum_psi_nYZ/valid_pts;
  return I;
}
