#pragma once
#include "mutual_information.hpp"


// ============================================================
// MI with prebuilt X marginal
//
// Runs the KSG MI estimator for MI(X, Y_data) reusing an X
// marginal that was already constructed by the caller.
// sa_X is non-null when dx==1 (sorted-array path).
// idx_X is non-null when dx>1  (kd-tree path).
// The joint XY tree and Y marginal are always built fresh.
// ============================================================
template<typename T>
T mi_with_prebuilt_x(
    T* X, T* Y_data, int k, int dx, int dy, int N,
    const SortedArray1D<T>*  sa_X,
    const my_kd_tree_t<T>*  idx_X)
{
  if (N == 0) return T(0);
  int dxy = dx + dy;

  // --- Joint XY data + tree (always new since Y differs) ---
  std::vector<T> XY(N * dxy);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XY[i*dxy+j]    = X[i*dx+j];
    for (int j = 0; j < dy; j++) XY[i*dxy+dx+j] = Y_data[i*dy+j];
  }
  PointCloud cloud_XY;
  cloud_XY.N = N; cloud_XY.dim = dxy; cloud_XY.pts = XY.data();
  my_kd_tree_t<T> index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));
  index_XY.buildIndex();

  // --- Y marginal: sorted array for 1D, kd-tree for >1D ---
  SortedArray1D<T> sa_Y;
  std::unique_ptr<my_kd_tree_t<T>> idx_Y;
  PointCloud cloud_Y;
  if (dy == 1) {
    sa_Y.build(Y_data, N);
  } else {
    cloud_Y.N = N; cloud_Y.dim = dy; cloud_Y.pts = Y_data;
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
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(0, false));
    T eps = out_dist[k];

    int cx;
    if (dx == 1) {
      cx = sa_X->countStrict(X[i], eps, i);
    } else {
      StrictRadiusCountSet<T> rs(eps, (size_t)i);
      idx_X->findNeighbors(rs, &X[i*dx], SearchParameters(0, false));
      cx = (int)rs.count;
    }

    int cy;
    if (dy == 1) {
      cy = sa_Y.countStrict(Y_data[i], eps, i);
    } else {
      StrictRadiusCountSet<T> rs(eps, (size_t)i);
      idx_Y->findNeighbors(rs, &Y_data[i*dy], SearchParameters(0, false));
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
// Information flow  dI/dt ≈ (MI(X, Y_shifted) − MI(X, Y)) / dt
//
// X marginal is built once and shared between both MI calls,
// avoiding the duplicate O(Neff log Neff) construction present
// when mutual_info() is called twice independently.
// ============================================================
template<typename T>
T information_flow(T* X, T* Y, int tau, T dt, int k, int dx, int dy, int N) {
  if (N <= tau) return T(0);
  int Neff = N - tau;
  T*  Z    = Y + tau * dy;  // pointer to Y[tau:], no copy

  // --- Build X marginal once ---
  SortedArray1D<T> sa_X;
  std::unique_ptr<my_kd_tree_t<T>> idx_X;
  PointCloud cloud_X;
  if (dx == 1) {
    sa_X.build(X, Neff);
  } else {
    cloud_X.N = Neff; cloud_X.dim = dx; cloud_X.pts = X;
    idx_X = std::make_unique<my_kd_tree_t<T>>(dx, cloud_X, KDTreeSingleIndexAdaptorParams(10));
    idx_X->buildIndex();
  }

  const SortedArray1D<T>*  sa_X_ptr  = (dx == 1)  ? &sa_X         : nullptr;
  const my_kd_tree_t<T>*   idx_X_ptr = (dx != 1)  ? idx_X.get()   : nullptr;

  T Ilag = mi_with_prebuilt_x(X, Z, k, dx, dy, Neff, sa_X_ptr, idx_X_ptr);
  T I    = mi_with_prebuilt_x(X, Y, k, dx, dy, Neff, sa_X_ptr, idx_X_ptr);
  return (Ilag - I) / dt;
}
