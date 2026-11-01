#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include <memory>
#include <omp.h>
#include "mutual_information.hpp"

namespace py = pybind11;


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

  std::vector<T> XY(N * dxy);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XY[i*dxy+j]    = X[i*dx+j];
    for (int j = 0; j < dy; j++) XY[i*dxy+dx+j] = Y_data[i*dy+j];
  }
  PointCloud cloud_XY;
  cloud_XY.N = N; cloud_XY.dim = dxy; cloud_XY.pts = XY.data();
  my_kd_tree_t<T> index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));
  index_XY.buildIndex();

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

    // Every point contributes, including n_x = 0 or n_y = 0 (legitimate for k = 1);
    // only degenerate points with eps == 0 (duplicates) are skipped.
    if (eps > 0) {
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
  T*  Z    = Y + tau * dy;

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


// ============================================================
// Point set for strict Chebyshev radius counts
//
// Owns a row-major copy of the concatenated blocks, e.g. (X_i, Z_i).
// 1D spaces use the sorted array, >1D spaces a kd-tree.  The PointCloud
// is heap-allocated so that the kd-tree's reference to it survives moves.
// ============================================================
template<typename T>
struct CountSpace {
  int n = 0, dim = 0;
  std::vector<T> pts;
  SortedArray1D<T> sa;
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<my_kd_tree_t<T>> tree;

  // blocks: {pointer to N x d row-major data, d}
  void build(const std::vector<std::pair<const T*, int>>& blocks, int N) {
    n = N; dim = 0;
    for (const auto& b : blocks) dim += b.second;
    pts.resize((size_t)N * dim);
    for (int i = 0; i < N; i++) {
      int off = 0;
      for (const auto& b : blocks) {
        for (int j = 0; j < b.second; j++)
          pts[(size_t)i*dim+off+j] = b.first[(size_t)i*b.second+j];
        off += b.second;
      }
    }
    if (dim == 1) {
      sa.build(pts.data(), N);
    } else {
      cloud = std::make_unique<PointCloud>();
      cloud->N = N; cloud->dim = dim; cloud->pts = pts.data();
      tree = std::make_unique<my_kd_tree_t<T>>(dim, *cloud, KDTreeSingleIndexAdaptorParams(10));
      tree->buildIndex();
    }
  }

  const T* point(int i) const { return &pts[(size_t)i * dim]; }

  // Number of points j != i with max_d |p_j - p_i| < eps.
  int count(int i, T eps) const {
    if (dim == 1) return sa.countStrict(pts[i], eps, i);
    StrictRadiusCountSet<T> rs(eps, (size_t)i);
    tree->findNeighbors(rs, point(i), SearchParameters(0, false));
    return (int)rs.count;
  }
};


// ============================================================
// CMI with prebuilt marginal spaces
//
// KSG (Frenzel-Pompe) estimate of I(X; Y | Z) reusing count spaces built
// by the caller with rows sZ = (Z_i), sXZ = (X_i, Z_i), sYZ = (Y_i, Z_i).
// The joint XYZ points are assembled from sXZ and the Y part of sYZ, so
// the joint space is consistent with the marginals by construction.
// ============================================================
template<typename T>
T cmi_with_prebuilt(const CountSpace<T>& sZ, const CountSpace<T>& sXZ,
                    const CountSpace<T>& sYZ, int k)
{
  const int N = sZ.n;
  if (sXZ.n != N || sYZ.n != N)
    throw std::runtime_error("cmi_with_prebuilt: count spaces differ in length");
  if (N <= k) return std::numeric_limits<T>::quiet_NaN();
  const int dxz  = sXZ.dim;
  const int dy   = sYZ.dim - sZ.dim;
  const int dxyz = dxz + dy;

  std::vector<T> XYZ((size_t)N * dxyz);
  for (int i = 0; i < N; i++) {
    const T* xz = sXZ.point(i);
    const T* yz = sYZ.point(i);
    for (int j = 0; j < dxz; j++) XYZ[(size_t)i*dxyz+j]     = xz[j];
    for (int j = 0; j < dy;  j++) XYZ[(size_t)i*dxyz+dxz+j] = yz[j];
  }
  PointCloud cloud_XYZ;
  cloud_XYZ.N = N; cloud_XYZ.dim = dxyz; cloud_XYZ.pts = XYZ.data();
  my_kd_tree_t<T> index_XYZ(dxyz, cloud_XYZ, KDTreeSingleIndexAdaptorParams(10));

  T   sum_psi_nZ = 0, sum_psi_nXZ = 0, sum_psi_nYZ = 0;
  int valid_pts  = 0;
  #pragma omp parallel reduction(+:sum_psi_nZ, sum_psi_nXZ, sum_psi_nYZ, valid_pts)
  {
  std::vector<size_t> ret_index(k+1);
  std::vector<T>      out_dist(k+1);
  #pragma omp for schedule(static)
  for (int i = 0; i < N; i++) {
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XYZ.findNeighbors(resultSet, &XYZ[(size_t)i * dxyz], SearchParameters(0, false));
    T eps = out_dist[k];

    // only degenerate points with eps == 0 (duplicates) are skipped
    if (eps > 0) {
      sum_psi_nZ  += digamma(sZ.count(i, eps)  + 1.0);
      sum_psi_nXZ += digamma(sXZ.count(i, eps) + 1.0);
      sum_psi_nYZ += digamma(sYZ.count(i, eps) + 1.0);
      ++valid_pts;
    }
  }
  }
  T I = T(0);
  if (valid_pts > 0)
    I = digamma(k) + sum_psi_nZ/valid_pts - sum_psi_nXZ/valid_pts - sum_psi_nYZ/valid_pts;
  return I;
}


// ============================================================
// Conditional information flow
//   dI_{X->Y|Z} ≈ (I(X_t; Y_{t+tau} | Z_t) − I(X_t; Y_t | Z_t)) / dt
//
// The Z and XZ spaces are shared between both CMI calls.
// ============================================================
template<typename T>
T conditional_information_flow(const T* X, const T* Y, const T* Z, int tau, T dt, int k,
                               int dx, int dy, int dz, int N) {
  if (N <= tau) return T(0);
  int Neff = N - tau;

  CountSpace<T> sZ, sXZ, sYZ, sYlagZ;
  sZ.build({{Z, dz}}, Neff);
  sXZ.build({{X, dx}, {Z, dz}}, Neff);
  sYZ.build({{Y, dy}, {Z, dz}}, Neff);
  sYlagZ.build({{Y + (size_t)tau * dy, dy}, {Z, dz}}, Neff);

  T Ilag = cmi_with_prebuilt(sZ, sXZ, sYlagZ, k);
  T I    = cmi_with_prebuilt(sZ, sXZ, sYZ,    k);
  return (Ilag - I) / dt;
}


// ============================================================
// Full causal map: no mask  →  (IF, Leak, dI)
// ============================================================
py::object information_flow_causal_map_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::array_t<int,    py::array::c_style> tau_obj,
  double dt=1.e0, int k=5, int n_threads=1)
{
  py::buffer_info info     = X_obj.request();
  py::buffer_info info_tau = tau_obj.request();
  if (info.ndim != 2 && info.ndim != 3)
    throw std::runtime_error("X must be 2D (N, Nt) or 3D (N, Nt, dx)");
  if (info_tau.ndim != 1)
    throw std::runtime_error("tau must be 1D (N)");
  if (info.itemsize != sizeof(double))
    throw std::runtime_error("X must be float64");
  if (info_tau.itemsize != sizeof(int))
    throw std::runtime_error("tau must be int32");

  py::ssize_t N  = info.shape[0];
  int Nt = static_cast<int>(info.shape[1]);
  int dx = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
  if (info_tau.shape[0] != N)
    throw std::runtime_error("tau length must equal N");

  double *X       = static_cast<double*>(info.ptr);
  int    *tau_arr = static_cast<int*>(info_tau.ptr);

  py::array_t<double>   IF_map({N,N});
  py::array_t<double>   dI_map({N,N});
  py::array_t<double> Leak_map({N,N});
  double *IF   = static_cast<double*>(IF_map.request().ptr);
  double *dI   = static_cast<double*>(dI_map.request().ptr);
  double *Leak = static_cast<double*>(Leak_map.request().ptr);

  py::ssize_t num_pairs = N * (N - 1) / 2;
  std::vector<std::pair<int,int>> pairs;
  pairs.reserve(num_pairs);
  for (int j = 0; j < N; ++j)
    for (int i = j + 1; i < N; ++i)
      pairs.push_back({j, i});

  for (int v = 0; v < N; ++v)
    IF[v*N+v] = dI[v*N+v] = Leak[v*N+v] = std::numeric_limits<double>::quiet_NaN();

  omp_set_num_threads(n_threads);

  if (dx == 1) {
    // Precompute per-variable sorted arrays (read-only during parallel loop, thread-safe).
    // sa_vars[v]     covers X[v][0:Neff]       — unlagged, used for I_ij, Ilag_ji, Ilag_ij
    // sa_lag_vars[v] covers X[v][tau:tau+Neff] — lagged,   used for Ilag_dI
    int tau  = tau_arr[0];   // tau_i == tau_j always
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
    for (py::ssize_t p = 0; p < num_pairs; ++p) {
      int j = pairs[p].first;
      int i = pairs[p].second;
      double* xi = X + i * Nt;
      double* xj = X + j * Nt;

      if (Neff <= 0) {
        IF[j*N+i] = IF[i*N+j] = dI[j*N+i] = dI[i*N+j] =
          Leak[j*N+i] = Leak[i*N+j] = std::numeric_limits<double>::quiet_NaN();
        continue;
      }

      double I_ij    = mi_with_prebuilt_x(xi, xj,       k, 1, 1, Neff, &sa_vars[i],     no_kd);
      double Ilag_ji = mi_with_prebuilt_x(xi, xj + tau, k, 1, 1, Neff, &sa_vars[i],     no_kd);
      double Ilag_ij = mi_with_prebuilt_x(xj, xi + tau, k, 1, 1, Neff, &sa_vars[j],     no_kd);
      double Ilag_dI = mi_with_prebuilt_x(xi + tau, xj + tau, k, 1, 1, Neff, &sa_lag_vars[i], no_kd);

      double if_ji    = (Ilag_ji - I_ij) / dt;
      double if_ij    = (Ilag_ij - I_ij) / dt;
      double di_val   = (Ilag_dI - I_ij) / dt;
      double leak_val = di_val - if_ji - if_ij;

      IF[j*N+i]   = if_ji;
      IF[i*N+j]   = if_ij;
      dI[j*N+i]   = dI[i*N+j]   = di_val;
      Leak[j*N+i] = Leak[i*N+j] = leak_val;
    }
  } else {
    // dx > 1: general path using mutual_info (builds all marginals per call)
    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t p = 0; p < num_pairs; ++p) {
      int j = pairs[p].first;
      int i = pairs[p].second;

      int    tau_i = tau_arr[i];
      int    tau_j = tau_arr[j];
      double *xi   = X + i * Nt * dx;
      double *xj   = X + j * Nt * dx;
      int Neff_i = Nt - tau_i;
      int Neff_j = Nt - tau_j;

      if (Neff_i <= 0 || Neff_j <= 0) {
        IF[j*N+i] = IF[i*N+j] = dI[j*N+i] = dI[i*N+j] =
          std::numeric_limits<double>::quiet_NaN();
        continue;
      }

      double I_i = mutual_info(&xi, &xj, k, dx, dx, Neff_i);

      double *yi    = xj + tau_i * dx;
      double Ilag_ji = mutual_info(&xi, &yi, k, dx, dx, Neff_i);
      IF[j*N+i] = (Ilag_ji - I_i) / dt;

      double I_j    = (tau_i == tau_j) ? I_i : mutual_info(&xj, &xi, k, dx, dx, Neff_j);
      double *yj    = xi + tau_j * dx;
      double Ilag_ij = mutual_info(&xj, &yj, k, dx, dx, Neff_j);
      IF[i*N+j] = (Ilag_ij - I_j) / dt;

      double *xi_s  = xi + tau_i * dx;
      double *xj_s  = xj + tau_i * dx;
      double Ilag_xy = mutual_info(&xi_s, &xj_s, k, dx, dx, Neff_i);
      dI[j*N+i] = dI[i*N+j] = (Ilag_xy - I_i) / dt;
    }

    // Leak: computed after IF and dI are fully written
    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t p = 0; p < num_pairs; ++p) {
      int j = pairs[p].first;
      int i = pairs[p].second;
      double leak_val = dI[j*N+i] - IF[j*N+i] - IF[i*N+j];
      Leak[j*N+i] = Leak[i*N+j] = leak_val;
    }
  }

  return py::make_tuple(IF_map, Leak_map, dI_map);
}


// ============================================================
// Full causal map: with mask  →  (IF, Leak, dI)
// ============================================================
py::object information_flow_causal_map_mask_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::object mask_obj,
  py::array_t<int, py::array::c_style> tau_obj,
  double dt=1.0, int k=5, int n_threads=1)
{
  py::buffer_info info_X = X_obj.request();
  if (info_X.ndim != 2)
    throw std::runtime_error("X must be 2D (Nx, Nt)");
  if (info_X.itemsize != sizeof(double))
    throw std::runtime_error("X must be float64");

  py::ssize_t Nx = info_X.shape[0];
  int Nt = static_cast<int>(info_X.shape[1]);
  double* X = static_cast<double*>(info_X.ptr);

  py::buffer_info info_tau = tau_obj.request();
  if (info_tau.ndim != 1 || info_tau.shape[0] != Nx)
    throw std::runtime_error("tau must be 1D with length Nx");
  if (info_tau.itemsize != sizeof(int))
    throw std::runtime_error("tau must be int32");
  int* tau_arr = static_cast<int*>(info_tau.ptr);

  std::vector<uint8_t> mask_default;
  bool* mask;
  py::array_t<bool, py::array::c_style> mask_arr;
  py::buffer_info info_mask;
  if (mask_obj.is_none()) {
    mask_default.assign(Nx * Nx, 1);
    for (int i = 0; i < Nx; ++i) mask_default[i * Nx + i] = 0;
    mask = reinterpret_cast<bool*>(mask_default.data());
  } else {
    mask_arr  = mask_obj.cast<py::array_t<bool, py::array::c_style>>();
    info_mask = mask_arr.request();
    if (info_mask.ndim != 2)
      throw std::runtime_error("mask must be 2D (Nx, Nx)");
    if (info_mask.itemsize != sizeof(bool))
      throw std::runtime_error("mask must be bool");
    if (info_mask.shape[0] != Nx || info_mask.shape[1] != Nx)
      throw std::runtime_error("mask must be (Nx, Nx)");
    mask = static_cast<bool*>(info_mask.ptr);
  }

  py::array_t<double>   IF_map({Nx, Nx});
  py::array_t<double>   dI_map({Nx, Nx});
  py::array_t<double> Leak_map({Nx, Nx});
  double* IF   = static_cast<double*>(IF_map.request().ptr);
  double* dI   = static_cast<double*>(dI_map.request().ptr);
  double* Leak = static_cast<double*>(Leak_map.request().ptr);
  std::fill(IF,   IF   + Nx * Nx, 0.0);
  std::fill(dI,   dI   + Nx * Nx, 0.0);
  std::fill(Leak, Leak + Nx * Nx, 0.0);

  for (int v = 0; v < Nx; ++v)
    IF[v*Nx+v] = dI[v*Nx+v] = Leak[v*Nx+v] = std::numeric_limits<double>::quiet_NaN();

  std::vector<std::pair<int,int>> pairs;
  for (int a = 0; a < Nx; ++a)
    for (int b = a + 1; b < Nx; ++b)
      if (mask[a * Nx + b] || mask[b * Nx + a])
        pairs.push_back({a, b});

  // Precompute per-variable sorted arrays (read-only during parallel loop, thread-safe).
  // sa_vars[v]     covers X[v][0:Neff]        — unlagged, used for I_ab, Ilag_ab, Ilag_ba
  // sa_lag_vars[v] covers X[v][tau:tau+Neff]  — lagged,   used for Ilag_dI
  int tau  = tau_arr[0];   // tau_a == tau_b always
  int Neff = Nt - tau;
  std::vector<SortedArray1D<double>> sa_vars(Nx), sa_lag_vars(Nx);
  if (Neff > 0) {
    for (int v = 0; v < Nx; ++v) {
      sa_vars[v].build(X + v * Nt, Neff);
      sa_lag_vars[v].build(X + v * Nt + tau, Neff);
    }
  }
  my_kd_tree_t<double>* const no_kd = nullptr;

  omp_set_num_threads(n_threads);
  const py::ssize_t num_pairs = static_cast<py::ssize_t>(pairs.size());
  #pragma omp parallel for schedule(dynamic)
  for (py::ssize_t p = 0; p < num_pairs; ++p) {
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

  return py::make_tuple(IF_map, Leak_map, dI_map);
}


// ============================================================
// IF-only causal map: no mask  →  IF_map
// ============================================================
py::array_t<double> information_flow_only_causal_map_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::array_t<int,    py::array::c_style> tau_obj,
  double dt=1.e0, int k=5, int n_threads=1)
{
  py::buffer_info info     = X_obj.request();
  py::buffer_info info_tau = tau_obj.request();
  if (info.ndim != 2 && info.ndim != 3)
    throw std::runtime_error("X must be 2D (N, Nt) or 3D (N, Nt, dx)");
  if (info_tau.ndim != 1)
    throw std::runtime_error("tau must be 1D (N)");
  if (info.itemsize != sizeof(double))
    throw std::runtime_error("X must be float64");
  if (info_tau.itemsize != sizeof(int))
    throw std::runtime_error("tau must be int32");

  py::ssize_t N  = info.shape[0];
  int Nt = static_cast<int>(info.shape[1]);
  int dx = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
  if (info_tau.shape[0] != N)
    throw std::runtime_error("tau length must equal N");

  double *X       = static_cast<double*>(info.ptr);
  int    *tau_arr = static_cast<int*>(info_tau.ptr);

  py::array_t<double> IF_map({N,N});
  double *IF = static_cast<double*>(IF_map.request().ptr);

  py::ssize_t num_pairs = N * (N - 1) / 2;
  std::vector<std::pair<int,int>> pairs;
  pairs.reserve(num_pairs);
  for (int j = 0; j < N; ++j)
    for (int i = j + 1; i < N; ++i)
      pairs.push_back({j, i});

  for (int v = 0; v < N; ++v)
    IF[v*N+v] = std::numeric_limits<double>::quiet_NaN();

  omp_set_num_threads(n_threads);
  #pragma omp parallel for schedule(dynamic)
  for (py::ssize_t p = 0; p < num_pairs; ++p) {
    int j = pairs[p].first;
    int i = pairs[p].second;

    int    tau_i = tau_arr[i];
    int    tau_j = tau_arr[j];
    double *xi   = X + i * Nt * dx;
    double *xj   = X + j * Nt * dx;
    int Neff_i = Nt - tau_i;
    int Neff_j = Nt - tau_j;

    if (Neff_i <= 0 || Neff_j <= 0) {
      IF[j*N+i] = IF[i*N+j] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    double I_i = mutual_info(&xi, &xj, k, dx, dx, Neff_i);

    double *yi    = xj + tau_i * dx;
    double Ilag_ji = mutual_info(&xi, &yi, k, dx, dx, Neff_i);
    IF[j*N+i] = (Ilag_ji - I_i) / dt;

    double I_j    = (tau_i == tau_j) ? I_i : mutual_info(&xj, &xi, k, dx, dx, Neff_j);
    double *yj    = xi + tau_j * dx;
    double Ilag_ij = mutual_info(&xj, &yj, k, dx, dx, Neff_j);
    IF[i*N+j] = (Ilag_ij - I_j) / dt;
  }

  return IF_map;
}


// ============================================================
// IF-only causal map: with mask  →  IF_map
// ============================================================
py::array_t<double> information_flow_only_causal_map_mask_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::object mask_obj,
  py::array_t<int, py::array::c_style> tau_obj,
  double dt=1.0, int k=5, int n_threads=1)
{
  py::buffer_info info_X = X_obj.request();
  if (info_X.ndim != 2)
    throw std::runtime_error("X must be 2D (Nx, Nt)");
  if (info_X.itemsize != sizeof(double))
    throw std::runtime_error("X must be float64");

  py::ssize_t Nx = info_X.shape[0];
  int Nt = static_cast<int>(info_X.shape[1]);
  double* X = static_cast<double*>(info_X.ptr);

  py::buffer_info info_tau = tau_obj.request();
  if (info_tau.ndim != 1 || info_tau.shape[0] != Nx)
    throw std::runtime_error("tau must be 1D with length Nx");
  if (info_tau.itemsize != sizeof(int))
    throw std::runtime_error("tau must be int32");
  int* tau_arr = static_cast<int*>(info_tau.ptr);

  std::vector<uint8_t> mask_default;
  bool* mask;
  py::array_t<bool, py::array::c_style> mask_arr;
  py::buffer_info info_mask;
  if (mask_obj.is_none()) {
    mask_default.assign(Nx * Nx, 1);
    for (int i = 0; i < Nx; ++i) mask_default[i * Nx + i] = 0;
    mask = reinterpret_cast<bool*>(mask_default.data());
  } else {
    mask_arr  = mask_obj.cast<py::array_t<bool, py::array::c_style>>();
    info_mask = mask_arr.request();
    if (info_mask.ndim != 2)
      throw std::runtime_error("mask must be 2D (Nx, Nx)");
    if (info_mask.itemsize != sizeof(bool))
      throw std::runtime_error("mask must be bool");
    if (info_mask.shape[0] != Nx || info_mask.shape[1] != Nx)
      throw std::runtime_error("mask must be (Nx, Nx)");
    mask = static_cast<bool*>(info_mask.ptr);
  }

  py::array_t<double> IF_map({Nx, Nx});
  double* IF = static_cast<double*>(IF_map.request().ptr);
  std::fill(IF, IF + Nx * Nx, 0.0);

  for (int v = 0; v < Nx; ++v)
    IF[v*Nx+v] = std::numeric_limits<double>::quiet_NaN();

  std::vector<std::pair<int,int>> pairs;
  for (int a = 0; a < Nx; ++a)
    for (int b = a + 1; b < Nx; ++b)
      if (mask[a * Nx + b] || mask[b * Nx + a])
        pairs.push_back({a, b});

  // Precompute per-variable sorted arrays (read-only during the parallel loop).
  // Each variable v has a fixed Neff_v = Nt - tau_arr[v].  Building once here
  // eliminates the O(N log N) marginal rebuild that mutual_info() would otherwise
  // perform for every pair that variable v participates in.
  std::vector<SortedArray1D<double>> sa_vars(Nx);
  for (int v = 0; v < Nx; ++v) {
    int neff = Nt - tau_arr[v];
    if (neff > 0) sa_vars[v].build(X + v * Nt, neff);
  }
  my_kd_tree_t<double>* const no_kd = nullptr;  // dx==1: kd-tree path never used

  omp_set_num_threads(n_threads);
  const py::ssize_t num_pairs = static_cast<py::ssize_t>(pairs.size());
  #pragma omp parallel for schedule(dynamic)
  for (py::ssize_t p = 0; p < num_pairs; ++p) {
    int a = pairs[p].first;
    int b = pairs[p].second;

    int     tau  = tau_arr[a];   // tau_a == tau_b always
    double* xa   = X + a * Nt;
    double* xb   = X + b * Nt;
    int     Neff = Nt - tau;

    if (Neff <= 0) {
      IF[a*Nx+b] = IF[b*Nx+a] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    double I_ab    = mi_with_prebuilt_x(xb, xa,       k, 1, 1, Neff, &sa_vars[b], no_kd);
    double Ilag_ab = mi_with_prebuilt_x(xb, xa + tau, k, 1, 1, Neff, &sa_vars[b], no_kd);
    double Ilag_ba = mi_with_prebuilt_x(xa, xb + tau, k, 1, 1, Neff, &sa_vars[a], no_kd);
    IF[a*Nx+b] = (Ilag_ab - I_ab) / dt;
    IF[b*Nx+a] = (Ilag_ba - I_ab) / dt;
  }

  return IF_map;
}


// ============================================================
// Conditional causal map (optional mask)  →  IF_map or (IF, Leak, dI)
//
// Every MI of the unconditional map is replaced by a CMI given Z_t:
//   IF[a, b] = (I(x_b,t; x_a,t+tau | Z_t) − I(x_a,t; x_b,t | Z_t)) / dt
//   dI[a, b] = (I(x_a,t+tau; x_b,t+tau | Z_t) − I(x_a,t; x_b,t | Z_t)) / dt
//   Leak     = dI − IF[a, b] − IF[b, a]
// Z is held at time t in every term, so Leak keeps its unconditional
// meaning (the part of the joint change not split into the two flows).
// Per-variable spaces (x_v,t, Z_t) and (x_v,t+tau, Z_t) and the Z_t space
// are built once; each CMI then builds only its joint tree.
// ============================================================
py::object information_flow_causal_map_cond_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::object Z_obj,
  py::object mask_obj,
  int tau, double dt=1.0, int k=5, int n_threads=1, bool full=false)
{
  py::buffer_info info = X_obj.request();
  if (info.ndim != 2 && info.ndim != 3)
    throw std::runtime_error("X must be 2D (N, Nt) or 3D (N, Nt, dx)");
  if (info.itemsize != sizeof(double))
    throw std::runtime_error("X must be float64");
  py::ssize_t Nx = info.shape[0];
  int Nt = static_cast<int>(info.shape[1]);
  int dx = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
  double* X = static_cast<double*>(info.ptr);

  auto Z_arr = Z_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  py::buffer_info info_Z = Z_arr.request();
  if (info_Z.ndim != 1 && info_Z.ndim != 2)
    throw std::runtime_error("Z must be 1D (Nt) or 2D (Nt, dz)");
  if (info_Z.shape[0] != Nt)
    throw std::runtime_error("Z must have Nt rows (one per time step of X)");
  int dz = (info_Z.ndim == 2) ? static_cast<int>(info_Z.shape[1]) : 1;
  if (dz < 1)
    throw std::runtime_error("Z must have at least one column");
  const double* Z = static_cast<const double*>(info_Z.ptr);

  bool has_mask = !mask_obj.is_none();
  py::array_t<bool, py::array::c_style> mask_arr;
  const bool* mask = nullptr;
  if (has_mask) {
    mask_arr = mask_obj.cast<py::array_t<bool, py::array::c_style>>();
    py::buffer_info info_mask = mask_arr.request();
    if (info_mask.ndim != 2)
      throw std::runtime_error("mask must be 2D (Nx, Nx)");
    if (info_mask.itemsize != sizeof(bool))
      throw std::runtime_error("mask must be bool");
    if (info_mask.shape[0] != Nx || info_mask.shape[1] != Nx)
      throw std::runtime_error("mask must be (Nx, Nx)");
    mask = static_cast<const bool*>(info_mask.ptr);
  }

  py::array_t<double>   IF_map({Nx, Nx});
  py::array_t<double>   dI_map({Nx, Nx});
  py::array_t<double> Leak_map({Nx, Nx});
  double* IF   = static_cast<double*>(IF_map.request().ptr);
  double* dI   = static_cast<double*>(dI_map.request().ptr);
  double* Leak = static_cast<double*>(Leak_map.request().ptr);
  const double NaN = std::numeric_limits<double>::quiet_NaN();
  std::fill(IF,   IF   + Nx * Nx, 0.0);
  std::fill(dI,   dI   + Nx * Nx, 0.0);
  std::fill(Leak, Leak + Nx * Nx, 0.0);
  for (py::ssize_t v = 0; v < Nx; ++v)
    IF[v*Nx+v] = dI[v*Nx+v] = Leak[v*Nx+v] = NaN;

  std::vector<std::pair<int,int>> pairs;
  std::vector<char> used(Nx, 0);
  for (int a = 0; a < Nx; ++a)
    for (int b = a + 1; b < Nx; ++b)
      if (!has_mask || mask[a * Nx + b] || mask[b * Nx + a]) {
        pairs.push_back({a, b});
        used[a] = used[b] = 1;
      }
  const py::ssize_t num_pairs = static_cast<py::ssize_t>(pairs.size());

  int Neff = Nt - tau;
  if (Neff <= 0) {
    for (const auto& p : pairs) {
      int a = p.first, b = p.second;
      IF[a*Nx+b] = IF[b*Nx+a] = dI[a*Nx+b] = dI[b*Nx+a] =
        Leak[a*Nx+b] = Leak[b*Nx+a] = NaN;
    }
    if (full) return py::make_tuple(IF_map, Leak_map, dI_map);
    return IF_map;
  }

  omp_set_num_threads(n_threads);

  // sA[v] = (x_v,t, Z_t), sB[v] = (x_v,t+tau, Z_t), t = 0..Neff-1
  CountSpace<double> sZ;
  sZ.build({{Z, dz}}, Neff);
  std::vector<CountSpace<double>> sA(Nx), sB(Nx);
  std::vector<int> used_vars;
  for (int v = 0; v < Nx; ++v)
    if (used[v]) used_vars.push_back(v);
  const int num_used = static_cast<int>(used_vars.size());
  #pragma omp parallel for schedule(dynamic)
  for (int u = 0; u < num_used; ++u) {
    int v = used_vars[u];
    const double* xv = X + (size_t)v * Nt * dx;
    sA[v].build({{xv, dx}, {Z, dz}}, Neff);
    sB[v].build({{xv + (size_t)tau * dx, dx}, {Z, dz}}, Neff);
  }

  #pragma omp parallel for schedule(dynamic)
  for (py::ssize_t p = 0; p < num_pairs; ++p) {
    int a = pairs[p].first;
    int b = pairs[p].second;

    double I_ab    = cmi_with_prebuilt(sZ, sA[a], sA[b], k);
    double Ilag_ab = cmi_with_prebuilt(sZ, sA[b], sB[a], k);   // x_b,t vs x_a,t+tau
    double Ilag_ba = cmi_with_prebuilt(sZ, sA[a], sB[b], k);   // x_a,t vs x_b,t+tau
    double if_ab   = (Ilag_ab - I_ab) / dt;
    double if_ba   = (Ilag_ba - I_ab) / dt;
    IF[a*Nx+b] = if_ab;
    IF[b*Nx+a] = if_ba;

    if (full) {
      double Ilag_dI = cmi_with_prebuilt(sZ, sB[a], sB[b], k);
      double di_val  = (Ilag_dI - I_ab) / dt;
      dI[a*Nx+b]   = dI[b*Nx+a]   = di_val;
      Leak[a*Nx+b] = Leak[b*Nx+a] = di_val - if_ab - if_ba;
    }
  }

  if (full) return py::make_tuple(IF_map, Leak_map, dI_map);
  return IF_map;
}


// ============================================================
// Unified dispatcher — routes based on mask and full flag
// ============================================================
py::object information_flow_causal_map_dispatcher(
    py::array_t<double, py::array::c_style> X_obj,
    py::object tau_obj,
    py::object mask_obj,
    double dt, int k, int n_threads,
    bool full,
    py::object Z_obj)
{
  // Broadcast scalar tau to a per-variable array
  py::array_t<int, py::array::c_style> tau_arr;
  if (py::isinstance<py::array>(tau_obj)) {
    tau_arr = tau_obj.cast<py::array_t<int, py::array::c_style>>();
  } else {
    int tau_val = tau_obj.cast<int>();
    int Nx = static_cast<int>(X_obj.request().shape[0]);
    tau_arr = py::array_t<int>({Nx});
    auto buf = tau_arr.mutable_unchecked<1>();
    for (int i = 0; i < Nx; ++i) buf(i) = tau_val;
  }

  // Every code path assumes one common delay (several read only tau[0]); reject
  // per-variable delays instead of silently returning path-dependent results.
  {
    auto t = tau_arr.unchecked<1>();
    for (py::ssize_t i = 1; i < t.shape(0); ++i)
      if (t(i) != t(0))
        throw std::invalid_argument("information_flow_causal_map requires the same tau for all variables");
    if (t.shape(0) > 0 && t(0) < 0)
      throw std::invalid_argument("tau must be non-negative");
  }

  if (!Z_obj.is_none()) {
    int tau = (tau_arr.size() > 0) ? tau_arr.at(0) : 0;
    return information_flow_causal_map_cond_wrapper(X_obj, Z_obj, mask_obj, tau, dt, k, n_threads, full);
  }

  if (full) {
    if (mask_obj.is_none())
      return information_flow_causal_map_wrapper(X_obj, tau_arr, dt, k, n_threads);
    else
      return information_flow_causal_map_mask_wrapper(X_obj, mask_obj, tau_arr, dt, k, n_threads);
  } else {
    if (mask_obj.is_none())
      return information_flow_only_causal_map_wrapper(X_obj, tau_arr, dt, k, n_threads);
    else
      return information_flow_only_causal_map_mask_wrapper(X_obj, mask_obj, tau_arr, dt, k, n_threads);
  }
}
