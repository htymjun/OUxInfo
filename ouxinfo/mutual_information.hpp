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
#include <omp.h>


using namespace nanoflann;
using boost::math::digamma;

template <typename T>
using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
  Chebyshev_Adaptor<T, PointCloud>,
  PointCloud,
  -1,
  size_t
>;


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

  // Count points strictly within radius of v, excluding self (original index = self).
  int countStrict(T v, T eps, int self) const {
    int lo = (int)(std::upper_bound(vals.begin(), vals.end(), v - eps) - vals.begin());
    int hi = (int)(std::lower_bound(vals.begin(), vals.end(), v + eps) - vals.begin());
    return hi - lo - 1;  // -1 excludes self
  }

  // Count with Theiler-window exclusion.
  // Subtract points j with |j - self| <= thei that happen to fall within the radius.
  int countTheiler(T v, T eps, int self, int thei, int N) const {
    int lo  = (int)(std::upper_bound(vals.begin(), vals.end(), v - eps) - vals.begin());
    int hi  = (int)(std::lower_bound(vals.begin(), vals.end(), v + eps) - vals.begin());
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
  if (N == 0) return T(0);
  T *X = *X_ptr;
  T *Y = *Y_ptr;
  int dxy = dx + dy;

  // --- Joint XY data ---
  std::vector<T> XY(N * dxy);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XY[i*dxy+j]    = X[i*dx+j];
    for (int j = 0; j < dy; j++) XY[i*dxy+dx+j] = Y[i*dy+j];
  }

  // --- Joint XY kd-tree (always needed) ---
  PointCloud cloud_XY;
  cloud_XY.N = N; cloud_XY.dim = dxy; cloud_XY.pts = XY.data();
  my_kd_tree_t<T> index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));
  index_XY.buildIndex();

  // --- X marginal: sorted array for 1D, kd-tree for >1D ---
  SortedArray1D<T> sa_X;
  std::unique_ptr<my_kd_tree_t<T>> idx_X;
  PointCloud cloud_X;
  if (dx == 1) {
    sa_X.build(X, N);
  } else {
    cloud_X.N = N; cloud_X.dim = dx; cloud_X.pts = X;
    idx_X = std::make_unique<my_kd_tree_t<T>>(dx, cloud_X, KDTreeSingleIndexAdaptorParams(10));
    idx_X->buildIndex();
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
    idx_Y->buildIndex();
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
    index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(0, false));
    T eps = out_dist[k];

    // --- X marginal radius count ---
    int cx;
    if (dx == 1) {
      cx = sa_X.countStrict(X[i], eps, i);
    } else {
      StrictRadiusCountSet<T> rs(eps, (size_t)i);
      idx_X->findNeighbors(rs, &X[i*dx], SearchParameters(0, false));
      cx = (int)rs.count;
    }

    // --- Y marginal radius count ---
    int cy;
    if (dy == 1) {
      cy = sa_Y.countStrict(Y[i], eps, i);
    } else {
      StrictRadiusCountSet<T> rs(eps, (size_t)i);
      idx_Y->findNeighbors(rs, &Y[i*dy], SearchParameters(0, false));
      cy = (int)rs.count;
    }

    if (cx > 0 && cy > 0) {
      sum_psi_nX += digamma(cx + 1.0);
      sum_psi_nY += digamma(cy + 1.0);
      ++valid_pts;
    }
  }
  }
  T I = T(0);
  if (valid_pts > 0)
    I = digamma(k) - sum_psi_nX/valid_pts - sum_psi_nY/valid_pts + digamma(N);
  return I;
}


// ============================================================
// mutual information with Theiler window
// ============================================================
template<typename T>
T mutual_info_Thei(T **X_ptr, T **Y_ptr, int k, int dx, int dy, int N, int Thei) {
  if (N == 0) return T(0);
  T *X = *X_ptr;
  T *Y = *Y_ptr;
  int dxy = dx + dy;

  // --- Joint XY data ---
  std::vector<T> XY(N * dxy);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XY[i*dxy+j]    = X[i*dx+j];
    for (int j = 0; j < dy; j++) XY[i*dxy+dx+j] = Y[i*dy+j];
  }

  // --- Joint XY kd-tree ---
  PointCloud cloud_XY;
  cloud_XY.N = N; cloud_XY.dim = dxy; cloud_XY.pts = XY.data();
  my_kd_tree_t<T> index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));
  index_XY.buildIndex();

  // --- X marginal ---
  SortedArray1D<T> sa_X;
  std::unique_ptr<my_kd_tree_t<T>> idx_X;
  PointCloud cloud_X;
  if (dx == 1) {
    sa_X.build(X, N);
  } else {
    cloud_X.N = N; cloud_X.dim = dx; cloud_X.pts = X;
    idx_X = std::make_unique<my_kd_tree_t<T>>(dx, cloud_X, KDTreeSingleIndexAdaptorParams(10));
    idx_X->buildIndex();
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
    idx_Y->buildIndex();
  }

  size_t num_search = std::min((size_t)N, (size_t)(k + 2 * Thei + 1));

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
    size_t num_results = index_XY.knnSearch(&XY[i*dxy], num_search,
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
      idx_X->radiusSearch(&X[i*dx], eps, matches, nanoflann::SearchParameters(0, false));
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
      idx_Y->radiusSearch(&Y[i*dy], eps, matches, nanoflann::SearchParameters(0, false));
      cy = 0;
      for (const auto& m : matches)
        if (std::abs((int)m.first - i) > Thei && m.second < eps) ++cy;
    }

    if (cx > 0 && cy > 0) {
      sum_psi_nX += digamma(cx + 1.0);
      sum_psi_nY += digamma(cy + 1.0);
      sum_psi_nN += digamma(nN_i + 1.0);
      ++valid_pts;
    }
  }
  }
  T I = T(0);
  if (valid_pts > 0)
    I = digamma(k) - sum_psi_nX/valid_pts - sum_psi_nY/valid_pts + sum_psi_nN/valid_pts;
  return I;
}


// ============================================================
// conditional mutual information I(X;Y|Z)
// ============================================================
template<typename T>
T conditional_mutual_info(T **x_ptr, T **y_ptr, T **z_ptr, int k, int dx, int dy, int dz, int N) {
  T *X = *x_ptr;
  T *Y = *y_ptr;
  T *Z = *z_ptr;
  int dxyz = dx + dy + dz; int dyz = dy + dz; int dxz = dx + dz;

  // --- Joint data XYZ, YZ, XZ ---
  std::vector<T> XYZ(N * dxyz), YZ(N * dyz), XZ(N * dxz);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XYZ[i*dxyz+j]       = X[i*dx+j];
    for (int j = 0; j < dy; j++) XYZ[i*dxyz+dx+j]    = Y[i*dy+j];
    for (int j = 0; j < dz; j++) XYZ[i*dxyz+dx+dy+j] = Z[i*dz+j];
    for (int j = 0; j < dy; j++) YZ[i*dyz+j]          = Y[i*dy+j];
    for (int j = 0; j < dz; j++) YZ[i*dyz+dy+j]       = Z[i*dz+j];
    for (int j = 0; j < dx; j++) XZ[i*dxz+j]          = X[i*dx+j];
    for (int j = 0; j < dz; j++) XZ[i*dxz+dx+j]       = Z[i*dz+j];
  }

  // --- XYZ kd-tree (always needed) ---
  PointCloud cloud_XYZ;
  cloud_XYZ.N = N; cloud_XYZ.dim = dxyz; cloud_XYZ.pts = XYZ.data();
  my_kd_tree_t<T> index_XYZ(dxyz, cloud_XYZ, KDTreeSingleIndexAdaptorParams(10));
  index_XYZ.buildIndex();

  // --- Z marginal: sorted array for 1D, kd-tree for >1D ---
  SortedArray1D<T> sa_Z;
  std::unique_ptr<my_kd_tree_t<T>> idx_Z;
  PointCloud cloud_Z;
  if (dz == 1) {
    sa_Z.build(Z, N);
  } else {
    cloud_Z.N = N; cloud_Z.dim = dz; cloud_Z.pts = Z;
    idx_Z = std::make_unique<my_kd_tree_t<T>>(dz, cloud_Z, KDTreeSingleIndexAdaptorParams(10));
    idx_Z->buildIndex();
  }

  // --- YZ and XZ always use kd-trees (joint dimension > 1D in typical use) ---
  PointCloud cloud_YZ, cloud_XZ;
  cloud_YZ.N = N; cloud_YZ.dim = dyz; cloud_YZ.pts = YZ.data();
  cloud_XZ.N = N; cloud_XZ.dim = dxz; cloud_XZ.pts = XZ.data();
  my_kd_tree_t<T> index_YZ(dyz, cloud_YZ, KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t<T> index_XZ(dxz, cloud_XZ, KDTreeSingleIndexAdaptorParams(10));
  index_YZ.buildIndex();
  index_XZ.buildIndex();

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
    index_XYZ.findNeighbors(resultSet, &XYZ[i * dxyz], SearchParameters(0, false));
    T eps = out_dist[k];

    // --- Z marginal radius count ---
    int cz;
    if (dz == 1) {
      cz = sa_Z.countStrict(Z[i], eps, i);
    } else {
      StrictRadiusCountSet<T> rs(eps, (size_t)i);
      idx_Z->findNeighbors(rs, &Z[i*dz], SearchParameters(0, false));
      cz = (int)rs.count;
    }

    // --- YZ and XZ radius counts ---
    StrictRadiusCountSet<T> rs_YZ(eps, (size_t)i), rs_XZ(eps, (size_t)i);
    index_YZ.findNeighbors(rs_YZ, &YZ[i*dyz], SearchParameters(0, false));
    index_XZ.findNeighbors(rs_XZ, &XZ[i*dxz], SearchParameters(0, false));
    int cyz = (int)rs_YZ.count, cxz = (int)rs_XZ.count;

    if (cz > 0 && cyz > 0 && cxz > 0) {
      sum_psi_nZ  += digamma(cz  + 1.0);
      sum_psi_nYZ += digamma(cyz + 1.0);
      sum_psi_nXZ += digamma(cxz + 1.0);
      ++valid_pts;
    }
  }
  }
  T I = T(0);
  if (valid_pts > 0)
    I = digamma(k) + sum_psi_nZ/valid_pts - sum_psi_nXZ/valid_pts - sum_psi_nYZ/valid_pts;
  return I;
}
