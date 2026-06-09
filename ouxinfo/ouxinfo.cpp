#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include "shannon_entropy.hpp"
#include "kullback_leibler_divergence.hpp"
#include "mutual_information.hpp"
#include "information_flow.hpp"


namespace py = pybind11;


double shannon_entropy_wrapper(py::array_t<double, py::array::c_style> x_obj, int k=5){
  /*
  Parameters
  ----------
  x : ndarray (N, dim)
  k : int, optional
      Number of nearest neighbors.
  Returns
  -------
  double
    Shannon entropy.
  */
  py::buffer_info info = x_obj.request();
  if (info.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info.itemsize != sizeof(double)) {
    throw std::runtime_error("Expected float64");
  }
  double *x = static_cast<double*>(info.ptr);
  int N = static_cast<int>(info.shape[0]);
  int d = static_cast<int>(info.shape[1]);
  return shannon_entropy(&x, k, d, N);
}


double KL_div_wrapper(py::array_t<double, py::array::c_style> x_obj, 
                      py::array_t<double, py::array::c_style> y_obj, int k=5){
  /*
  Parameters
  ----------
  x : ndarray (N, dim)
  y : ndarray (N, dim)
  k : int, optional
      Number of nearest neighbors.
  Returns
  -------
  double
    KL divergence.
  */
  py::buffer_info info_x = x_obj.request();
  py::buffer_info info_y = y_obj.request();
  if (info_x.ndim != 2 || info_y.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info_x.itemsize != sizeof(double) || info_y.itemsize != sizeof(double)) {
    throw std::runtime_error("Expected float64");
  }
  double *x = static_cast<double*>(info_x.ptr);
  double *y = static_cast<double*>(info_y.ptr);
  int N = static_cast<int>(info_x.shape[0]);
  int M = static_cast<int>(info_y.shape[0]);
  int d = static_cast<int>(info_x.shape[1]);
  return KL_div(&x, &y, k, d, N, M);
}


double mutual_info_wrapper(py::array_t<double, py::array::c_style> x_obj, 
                           py::array_t<double, py::array::c_style> y_obj, int k=5, int Thei=0){
  /*
  Parameters
  ----------
  x    : ndarray (N, dim)
  y    : ndarray (N, dim)
  k    : int, optional
         Number of nearest neighbors.
  Thei : int, optional
         Length of Theiler window.
  Returns
  -------
  double
    mutual information.
  */
  py::buffer_info info_x = x_obj.request();
  py::buffer_info info_y = y_obj.request();
  if (info_x.ndim != 2 || info_y.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info_x.itemsize != sizeof(double) || info_y.itemsize != sizeof(double)) {
    throw std::runtime_error("Expected float64");
  }
  double *x = static_cast<double*>(info_x.ptr);
  double *y = static_cast<double*>(info_y.ptr);
  int N = static_cast<int>(info_x.shape[0]);
  int M = static_cast<int>(info_y.shape[0]);
  if (N != M) {
    throw std::runtime_error("Input argument must be the same length");
  }
  int dx = static_cast<int>(info_x.shape[1]);
  int dy = static_cast<int>(info_y.shape[1]);
  double I;
  if (Thei == 0) {
    I = mutual_info(&x, &y, k, dx, dy, N);
  }
  else { 
    I = mutual_info_Thei(&x, &y, k, dx, dy, N, Thei);
  }
  return I;
}


double conditional_mutual_info_wrapper(py::array_t<double, py::array::c_style> x_obj,
                                       py::array_t<double, py::array::c_style> y_obj,
                                       py::array_t<double, py::array::c_style> z_obj, int k=5){
  /*
  Parameters
  ----------
  x : ndarray (N, dim)
  y : ndarray (N, dim)
  z : ndarray (N, dim)
  k : int, optional
      Number of nearest neighbors.
  Returns
  -------
  double
    conditional mutual information.
  */
  py::buffer_info info_x = x_obj.request();
  py::buffer_info info_y = y_obj.request();
  py::buffer_info info_z = z_obj.request();
  if (info_x.ndim != 2 || info_y.ndim != 2 || info_z.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info_x.itemsize != sizeof(double) || info_y.itemsize != sizeof(double) || info_z.itemsize != sizeof(double)) {
    throw std::runtime_error("Expected float64");
  }
  double *x = static_cast<double*>(info_x.ptr);
  double *y = static_cast<double*>(info_y.ptr);
  double *z = static_cast<double*>(info_z.ptr);
  int Nx = static_cast<int>(info_x.shape[0]);
  int Ny = static_cast<int>(info_y.shape[0]);
  int Nz = static_cast<int>(info_z.shape[0]);
  if (Nx != Ny || Ny != Nz || Nz != Nx) {
    throw std::runtime_error("Input argument must be the same length");
  }
  int dx = static_cast<int>(info_x.shape[1]);
  int dy = static_cast<int>(info_y.shape[1]);
  int dz = static_cast<int>(info_z.shape[1]);
  double I;
  I = conditional_mutual_info(&x, &y, &z, k, dx, dy, dz, Nx);
  return I;
}


double transfer_entropy_wrapper(py::array_t<double, py::array::c_style> x_obj, 
                                py::array_t<double, py::array::c_style> y_obj,
                                int tau=1, int m=1, int lag=1, double dt=1.e0, int k=5, int trial=0){
  /*
  Parameters
  ----------
  x     : ndarray (N, dim)
  y     : ndarray (N, dim)
  tau   : int, optional
          Length of time delay
  m     : int, optional
          Embedding dimension for y
  lag   : int, optional
          Time lag for embedding
  dt    : double, optional
          Physical time
  k     : int, optional
          Number of nearest neighbors.
  trial : int, optional
          The number of trials for surrogate analysis.
  Returns
  -------
  double
    transfer entropy.
  */
  py::buffer_info info_x = x_obj.request();
  py::buffer_info info_y = y_obj.request();
  if (info_x.ndim != 2 || info_y.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info_x.itemsize != sizeof(double) || info_y.itemsize != sizeof(double)) {
    throw std::runtime_error("Expected float64");
  }
  double *x = static_cast<double*>(info_x.ptr);
  double *y = static_cast<double*>(info_y.ptr);
  int N = static_cast<int>(info_x.shape[0]);
  int M = static_cast<int>(info_y.shape[0]);
  if (N != M) {
    throw std::runtime_error("Input argument must be the same length");
  }
  int dx = static_cast<int>(info_x.shape[1]);
  int dy = static_cast<int>(info_y.shape[1]);
  int i_start = (m - 1) * lag;
  int Nt = N - tau - i_start;
  if (Nt <= 0) {
    throw std::runtime_error("Not enough data points for the given tau, m, and lag.");
  }
  // Y history
  int dy_past = m * dy;
  std::vector<double> y_past_data(Nt * dy_past);
  double *y_past = y_past_data.data();
  for (int i = 0; i < Nt; ++i) {
    int current_time = i_start + i;
    for (int j = 0; j < m; ++j) {
      int past_time = current_time - j * lag;
      std::copy(y + past_time * dy,
                y + past_time * dy + dy,
                y_past + i * dy_past + j * dy);
    }
  }
  // offset x & z
  double *x_valid = x + i_start * dx;
  double *z_valid = y + (i_start + tau) * dy;
  double TE  = conditional_mutual_info(&x_valid, &z_valid, &y_past, k, dx, dy, dy_past, Nt); // I(X^n;Y^{n+tau}|Y^n)
  double TEs = 0.e0;
  if (trial > 0) {
    std::vector<int> idx(Nt);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<double> xs_data(Nt * dx);
    for (int itr = 0; itr < trial; itr++) {
      std::shuffle(idx.begin(), idx.end(), g);
      for (int i = 0; i < Nt; ++i) {
        int original_idx = idx[i];
        std::copy(x_valid + (original_idx * dx),
                  x_valid + (original_idx * dx) + dx,
                  xs_data.begin() + (i * dx));
      }
      double *xs_ptr = xs_data.data();
      TEs += conditional_mutual_info(&xs_ptr, &z_valid, &y_past, k, dx, dy, dy_past, Nt); // I(Xs^n:Y^{n+tau}|Y^n)
    }
    TEs /= static_cast<double>(trial);
  }
  return (TE - TEs) / dt;
}


double information_flow_wrapper(py::array_t<double, py::array::c_style> x_obj, 
                                py::array_t<double, py::array::c_style> y_obj,
                                int tau=1, double dt=1.e0, int k=5){
  /*
  Parameters
  ----------
  x   : ndarray (N, dim)
  y   : ndarray (N, dim)
  tau : int, optional
        Length of time delay
  dt  : double, optional
        Physical time
  k   : int, optional
        Number of nearest neighbors.
  Returns
  -------
  double
    information flow.
  */
  py::buffer_info info_x = x_obj.request();
  py::buffer_info info_y = y_obj.request();
  if (info_x.ndim != 2 || info_y.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info_x.itemsize != sizeof(double) || info_y.itemsize != sizeof(double)) {
    throw std::runtime_error("Expected float64");
  }
  double *x = static_cast<double*>(info_x.ptr);
  double *y = static_cast<double*>(info_y.ptr);
  int N = static_cast<int>(info_x.shape[0]);
  int M = static_cast<int>(info_y.shape[0]);
  if (N != M) {
    throw std::runtime_error("Input argument must be the same length");
  }
  int dx = static_cast<int>(info_x.shape[1]);
  int dy = static_cast<int>(info_y.shape[1]);
  return information_flow<double>(x, y, tau, dt, k, dx, dy, N);
}


py::array_t<double> transfer_entropy_causal_map_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::array_t<int,    py::array::c_style> tau_obj,
  int m=1, int lag=1, double dt=1.e0, int k=5, int trial=0, int n_threads=1){
  /*
  Parameters
  ----------
  X         : ndarray (N, Nt) or (N, Nt, dim)
  tau       : ndarray (N)
              Length of time delay
  m         : int, optional
              Embedding dimension for y
  lag       : int, optional
              Time lag for embedding
  dt        : double, optional
              Physical time
  k         : int, optional
              Number of nearest neighbors.
  trial     : int, optional
              The number of trials for surrogate analysis.
  n_threads : int, optional
              The number of threads used for OpenMP.
  Returns
  -------
  double
    transfer entropy causal map.
  */
  py::buffer_info info     = X_obj.request();
  py::buffer_info info_tau = tau_obj.request();
  // error messages
  if (info.ndim != 2 && info.ndim != 3)
    throw std::runtime_error("X must be 2D (N, Nt) or 3D (N, Nt, dx)");
  if (info_tau.ndim != 1)
    throw std::runtime_error("tau must be 1D (N)");
  if (info.itemsize != sizeof(double))
    throw std::runtime_error("X must be float64");
  if (info_tau.itemsize != sizeof(int))
    throw std::runtime_error("tau must be int32");
  // main
  int N  = static_cast<int>(info.shape[0]);
  int Nt = static_cast<int>(info.shape[1]);
  int dx = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
  if (info_tau.shape[0] != N)
    throw std::runtime_error("tau length must equal N");
  // pointer
  double *X = static_cast<double*>(info.ptr);
  int *tau_arr = static_cast<int*>(info_tau.ptr);
  py::array_t<double> TE_map({N,N});
  py::buffer_info info_out = TE_map.request();
  double *TE = static_cast<double*>(info_out.ptr);
  // History
  int dx_past = m * dx;
  int i_start = (m-1) * lag;
  // main loop
  omp_set_num_threads(n_threads);
  #pragma omp parallel
  {
    std::mt19937 rng(omp_get_thread_num() + 1234);
    std::vector<double> xj_past_data(Nt * dx_past);
    std::vector<double> xs_data(Nt * dx);
    std::vector<int> idx(Nt);
    #pragma omp for schedule(dynamic)
    for (int j = 0; j < N; j++) {
      double *xj = X + j * Nt * dx;
      for (int i = 0; i < N; i++) {
        if (i == j) {
          TE[j*N + i] = std::numeric_limits<double>::quiet_NaN();
          continue;
        }
        int tau = tau_arr[i];
        int Neff = Nt - tau - i_start;
        if (Neff <= 0) {
          TE[j*N+i] = std::numeric_limits<double>::quiet_NaN();
          continue;
        }
        // Target History
        double *xj_past = xj_past_data.data();
        for (int n = 0; n < Neff; ++n) {
          int current_time = i_start + n;
          for (int m_idx = 0; m_idx < m; ++m_idx) {
            int past_time = current_time - m_idx * lag;
            std::copy(xj + past_time * dx,
                      xj + past_time * dx + dx,
                      xj_past + n * dx_past + m_idx * dx);
          }
        }
        // offset
        double *xi = X + i * Nt * dx;
        double *xi_valid = xi + i_start * dx;
        double *z_valid  = xj + (i_start + tau) * dx;
        // I(X^n;Y^{n+tau}|Y^past)
        double TE_val = conditional_mutual_info(&xi_valid, &z_valid, &xj_past, k, dx, dx, dx_past, Neff);
        double TEs = 0.0;
        if (trial > 0) {
          std::iota(idx.begin(), idx.begin() + Neff, 0);
          for (int itr = 0; itr < trial; ++itr) {
            std::shuffle(idx.begin(), idx.begin() + Neff, rng);
            for (int t = 0; t < Neff; ++t) {
              int original_idx = idx[t];
              std::copy(xi_valid + (original_idx * dx),
                        xi_valid + (original_idx * dx) + dx,
                        xs_data.begin() + (t * dx));
            }
            double *xs_ptr = xs_data.data();
            // I(X^n;Y^{n+tau}|Y^past)
            TEs += conditional_mutual_info(&xs_ptr, &z_valid, &xj_past, k, dx, dx, dx_past, Neff);
          }
          TEs /= static_cast<double>(trial);
        }
        TE[j*N + i] = (TE_val - TEs) / dt;
      }
    }
  }
  return TE_map;
}


py::array_t<double> information_flow_causal_map_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::array_t<int,    py::array::c_style> tau_obj,
  double dt=1.e0, int k=5, int n_threads=1){
  /*
  Parameters
  ----------
  X         : ndarray (N, Nt) or (N, Nt, dim)
  tau       : ndarray (N)
              Length of time delay
  dt        : double, optional
              Physical time
  k         : int, optional
              Number of nearest neighbors.
  n_threads : int, optional
              The number of threads used for OpenMP.
  Returns
  -------
  double
    information flow causal map.
  */
  py::buffer_info info     = X_obj.request();
  py::buffer_info info_tau = tau_obj.request();
  // error messages
  if (info.ndim != 2 && info.ndim != 3)
    throw std::runtime_error("X must be 2D (N, Nt) or 3D (N, Nt, dx)");
  if (info_tau.ndim != 1)
    throw std::runtime_error("tau must be 1D (N)");
  if (info.itemsize != sizeof(double))
    throw std::runtime_error("X must be float64");
  if (info_tau.itemsize != sizeof(int))
    throw std::runtime_error("tau must be int32");
  // main
  int N  = static_cast<int>(info.shape[0]);
  int Nt = static_cast<int>(info.shape[1]);
  int dx = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
  if (info_tau.shape[0] != N)
    throw std::runtime_error("tau length must equal N");
  // pointer
  double *X = static_cast<double*>(info.ptr);
  int *tau_arr = static_cast<int*>(info_tau.ptr);
  py::array_t<double>   IF_map({N,N});
  py::array_t<double>   dI_map({N,N});
  py::array_t<double> Leak_map({N,N});
  py::buffer_info info_IF   = IF_map.request();
  py::buffer_info info_dI   = dI_map.request();
  py::buffer_info info_Leak = Leak_map.request();
  double *IF   = static_cast<double*>(info_IF.ptr);
  double *dI   = static_cast<double*>(info_dI.ptr);
  double *Leak = static_cast<double*>(info_Leak.ptr);
  // Enumerate unordered pairs {j,i} with j < i.  Processing each pair together
  // lets us reuse the base MI(Xi, Xj) for both IF directions and skip a redundant
  // call when tau_i == tau_j (20% fewer MI calls under uniform-tau configurations).
  int num_pairs = N * (N - 1) / 2;
  std::vector<std::pair<int,int>> pairs;
  pairs.reserve(num_pairs);
  for (int j = 0; j < N; ++j)
    for (int i = j + 1; i < N; ++i)
      pairs.push_back({j, i});

  // Diagonal: NaN
  for (int v = 0; v < N; ++v)
    IF[v*N+v] = dI[v*N+v] = Leak[v*N+v] = std::numeric_limits<double>::quiet_NaN();

  // main loop
  omp_set_num_threads(n_threads);
  #pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < num_pairs; ++p) {
    int j = pairs[p].first;
    int i = pairs[p].second;   // i > j

    int tau_i = tau_arr[i];
    int tau_j = tau_arr[j];
    double *xi = X + i * Nt * dx;
    double *xj = X + j * Nt * dx;
    int Neff_i = Nt - tau_i;
    int Neff_j = Nt - tau_j;

    if (Neff_i <= 0 || Neff_j <= 0) {
      IF[j*N+i] = IF[i*N+j] = dI[j*N+i] = dI[i*N+j] =
        std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    // --- Base MI: MI(Xi[:Neff_i], Xj[:Neff_i]) ---
    // Reused for IF[j,i], dI, and IF[i,j] when tau_i == tau_j.
    double I_i = mutual_info(&xi, &xj, k, dx, dx, Neff_i);

    // --- IF[j,i]: source i, target j ---
    double *yi = xj + tau_i * dx;   // Xj shifted by tau_i
    double Ilag_ji = mutual_info(&xi, &yi, k, dx, dx, Neff_i);
    IF[j*N+i] = (Ilag_ji - I_i) / dt;

    // --- IF[i,j]: source j, target i ---
    // MI(Xi,Xj)==MI(Xj,Xi) and Neff equal when tau_i==tau_j, so reuse I_i.
    double I_j = (tau_i == tau_j) ? I_i : mutual_info(&xj, &xi, k, dx, dx, Neff_j);
    double *yj = xi + tau_j * dx;   // Xi shifted by tau_j
    double Ilag_ij = mutual_info(&xj, &yj, k, dx, dx, Neff_j);
    IF[i*N+j] = (Ilag_ij - I_j) / dt;

    // --- dI[j,i]: MI(Xi[tau_i:], Xj[tau_i:]) ---
    double *xi_s = xi + tau_i * dx;
    double *xj_s = xj + tau_i * dx;
    double Ilag_xy = mutual_info(&xi_s, &xj_s, k, dx, dx, Neff_i);
    dI[j*N+i] = dI[i*N+j] = (Ilag_xy - I_i) / dt;
  }

  // Leak: computed after all IF and dI values are available
  #pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < num_pairs; ++p) {
    int j = pairs[p].first;
    int i = pairs[p].second;
    double leak_val = dI[j*N+i] - IF[j*N+i] - IF[i*N+j];
    Leak[j*N+i] = Leak[i*N+j] = leak_val;
  }

  return py::make_tuple(IF_map, Leak_map, dI_map);
}


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

  int Nx = static_cast<int>(info_X.shape[0]);
  int Nt = static_cast<int>(info_X.shape[1]);
  double* X = static_cast<double*>(info_X.ptr);

  py::buffer_info info_tau = tau_obj.request();
  if (info_tau.ndim != 1 || info_tau.shape[0] != Nx)
    throw std::runtime_error("tau must be 1D with length Nx");
  if (info_tau.itemsize != sizeof(int))
    throw std::runtime_error("tau must be int32");
  int* tau_arr = static_cast<int*>(info_tau.ptr);

  // mask=None → all off-diagonal pairs
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

  // Allocate three output arrays initialised to 0
  py::array_t<double>   IF_map({Nx, Nx});
  py::array_t<double>   dI_map({Nx, Nx});
  py::array_t<double> Leak_map({Nx, Nx});
  double* IF   = static_cast<double*>(IF_map.request().ptr);
  double* dI   = static_cast<double*>(dI_map.request().ptr);
  double* Leak = static_cast<double*>(Leak_map.request().ptr);
  std::fill(IF,   IF   + Nx * Nx, 0.0);
  std::fill(dI,   dI   + Nx * Nx, 0.0);
  std::fill(Leak, Leak + Nx * Nx, 0.0);

  // Diagonal: NaN (matches full wrapper convention)
  for (int v = 0; v < Nx; ++v)
    IF[v*Nx+v] = dI[v*Nx+v] = Leak[v*Nx+v] = std::numeric_limits<double>::quiet_NaN();

  // Collect unique unordered pairs {a, b} with a < b where either mask entry is True
  std::vector<std::pair<int,int>> pairs;
  for (int a = 0; a < Nx; ++a)
    for (int b = a + 1; b < Nx; ++b)
      if (mask[a * Nx + b] || mask[b * Nx + a])
        pairs.push_back({a, b});

  omp_set_num_threads(n_threads);
  #pragma omp parallel for schedule(dynamic)
  for (int p = 0; p < (int)pairs.size(); ++p) {
    int a = pairs[p].first;
    int b = pairs[p].second;  // b > a — mirrors j < i convention in full wrapper

    int tau_a = tau_arr[a];
    int tau_b = tau_arr[b];
    double* xa = X + a * Nt;
    double* xb = X + b * Nt;
    int Neff_a = Nt - tau_a;
    int Neff_b = Nt - tau_b;

    if (Neff_a <= 0 || Neff_b <= 0) {
      if (mask[a * Nx + b]) IF[a*Nx+b] = dI[a*Nx+b] = Leak[a*Nx+b] = std::numeric_limits<double>::quiet_NaN();
      if (mask[b * Nx + a]) IF[b*Nx+a] = dI[b*Nx+a] = Leak[b*Nx+a] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    // Base MI using tau_b (larger-index variable), mirrors I_i in full wrapper
    double I_b = mutual_info(&xb, &xa, k, 1, 1, Neff_b);

    // IF[a,b]: source = Xb → target = Xa  (mirrors IF[j,i] in full wrapper)
    double* ya = xa + tau_b;
    double Ilag_ab = mutual_info(&xb, &ya, k, 1, 1, Neff_b);
    double if_ab = (Ilag_ab - I_b) / dt;

    // IF[b,a]: source = Xa → target = Xb  (mirrors IF[i,j] in full wrapper)
    double I_a = (tau_a == tau_b) ? I_b : mutual_info(&xa, &xb, k, 1, 1, Neff_a);
    double* yb = xb + tau_a;
    double Ilag_ba = mutual_info(&xa, &yb, k, 1, 1, Neff_a);
    double if_ba = (Ilag_ba - I_a) / dt;

    // dI: symmetric, uses tau_b (larger-index variable), mirrors dI in full wrapper
    double* xb_s = xb + tau_b;
    double* xa_s = xa + tau_b;
    double Ilag_dI = mutual_info(&xb_s, &xa_s, k, 1, 1, Neff_b);
    double di_val = (Ilag_dI - I_b) / dt;

    double leak_val = di_val - if_ba - if_ab;

    // Store IF only for masked entries; dI and Leak symmetrically for masked entries
    if (mask[a * Nx + b]) { IF[a*Nx+b] = if_ab; dI[a*Nx+b] = di_val; Leak[a*Nx+b] = leak_val; }
    if (mask[b * Nx + a]) { IF[b*Nx+a] = if_ba; dI[b*Nx+a] = di_val; Leak[b*Nx+a] = leak_val; }
  }

  return py::make_tuple(IF_map, Leak_map, dI_map);
}


// ============================================================
// Unified dispatcher: routes to full or mask wrapper based on mask argument
// ============================================================
py::object information_flow_causal_map_dispatcher(
    py::array_t<double, py::array::c_style> X_obj,
    py::object tau_obj,
    py::object mask_obj,
    double dt, int k, int n_threads)
{
  if (mask_obj.is_none()) {
    // No mask: full computation (returns tuple of three arrays)
    // tau may be scalar int or int32 array; broadcast scalar to per-variable array
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
    return information_flow_causal_map_wrapper(X_obj, tau_arr, dt, k, n_threads);
  } else {
    // Mask provided: masked computation (returns tuple of three arrays)
    // tau may be scalar int or int32 array; broadcast scalar to per-variable array
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
    return information_flow_causal_map_mask_wrapper(X_obj, mask_obj, tau_arr, dt, k, n_threads);
  }
}


// ============================================================
// pybind11 module
// ============================================================
PYBIND11_MODULE(_core, m) {
  m.doc() = "High-performance Shannon entropy and information-theoretic estimators (C++ backend).";

  py::options options;
  options.disable_function_signatures();

  m.def("shannon_entropy", &shannon_entropy_wrapper,
        py::arg("X"), py::arg("k")=5,
        R"doc(Compute Shannon entropy using the Kozachenko-Leonenko estimator.

Parameters
----------
X : ndarray of shape (N, dim)
    Input data. Must be float64.
k : int, optional
    Number of nearest neighbors. Default 5.

Returns
-------
value : float
    Shannon entropy in nats.

Notes
-----
Uses the Kozachenko-Leonenko k-NN estimator with Chebyshev metric.
See: Kozachenko & Leonenko (1987), Kraskov et al. (2004).
)doc");
  m.def("KL_div", &KL_div_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("k")=5,
        R"doc(Compute Kullback-Leibler divergence using the Pérez-Cruz estimator.

Parameters
----------
X : ndarray of shape (N, dim)
    Samples from distribution X. Must be float64.
Y : ndarray of shape (M, dim)
    Samples from distribution Y. Must be float64.
k : int, optional
    Number of nearest neighbors. Default 5.

Returns
-------
value : float
    KL divergence D_KL(X || Y) in nats.

Notes
-----
Uses the k-NN estimator of Pérez-Cruz (2008).
)doc");
  m.def("mutual_info", &mutual_info_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("k")=5, py::arg("Thei")=0,
        R"doc(Compute mutual information using the Kraskov-Stögbauer-Grassberger (KSG) estimator.

Parameters
----------
X : ndarray of shape (N, dim_x)
    First dataset. Must be float64.
Y : ndarray of shape (N, dim_y)
    Second dataset. Must be float64.
k : int, optional
    Number of nearest neighbors. Default 5.
Thei : int, optional
    Length of Theiler window to exclude temporal neighbors. Default 0.

Returns
-------
value : float
    Mutual information I(X; Y) in nats.

Notes
-----
Uses KSG estimator type 1. See: Kraskov et al. (2004).
)doc");
  m.def("conditional_mutual_info", &conditional_mutual_info_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("k")=5,
        R"doc(Compute conditional mutual information using the KSG estimator.

Parameters
----------
X : ndarray of shape (N, dim_x)
    First dataset. Must be float64.
Y : ndarray of shape (N, dim_y)
    Second dataset. Must be float64.
Z : ndarray of shape (N, dim_z)
    Conditioning dataset. Must be float64.
k : int, optional
    Number of nearest neighbors. Default 5.

Returns
-------
value : float
    Conditional mutual information I(X; Y | Z) in nats.

Notes
-----
Uses KSG estimator type 1. See: Kraskov et al. (2004).
)doc");
  m.def("transfer_entropy", &transfer_entropy_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("tau")=1, py::arg("m")=1, py::arg("lag")=1,
        py::arg("dt")=1.e0, py::arg("k")=5, py::arg("trial")=0,
        R"doc(Compute transfer entropy from X to Y using the KSG estimator.

Parameters
----------
X : ndarray of shape (N, dim)
    Source time series. Must be float64.
Y : ndarray of shape (N, dim)
    Target time series. Must be float64.
tau : int, optional
    Time delay (in samples). Default 1.
m : int, optional
    Embedding dimension for Y. Default 1.
lag : int, optional
    Time lag for embedding. Default 1.
dt : float, optional
    Physical time step. Default 1.0.
k : int, optional
    Number of nearest neighbors. Default 5.
trial : int, optional
    Number of surrogate trials for significance testing. Default 0.

Returns
-------
value : float
    Transfer entropy TE(X -> Y) in nats per dt.

Notes
-----
Computed as conditional mutual information: TE = I(Y_{t+tau}; X_t^(m) | Y_t^(m)).
See: Schreiber (2000), Kraskov et al. (2004).
)doc");
  m.def("information_flow", &information_flow_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("tau")=1, py::arg("dt")=1.e0, py::arg("k")=5,
        R"doc(Compute information flow from X to Y using the KSG estimator.

Parameters
----------
X : ndarray of shape (N, dim)
    Source time series. Must be float64.
Y : ndarray of shape (N, dim)
    Target time series. Must be float64.
tau : int, optional
    Time delay (in samples). Default 1.
dt : float, optional
    Physical time step. Default 1.0.
k : int, optional
    Number of nearest neighbors. Default 5.

Returns
-------
value : float
    Information flow dI_X (rate of mutual information change due to X) in nats per dt.

Notes
-----
Based on the decomposition

    dI/dt = dI_X + dI_Y + Leak

See: Horowitz & Esposito (2014).
)doc");
  m.def("transfer_entropy_causal_map", &transfer_entropy_causal_map_wrapper,
        py::arg("X"), py::arg("tau"), py::arg("m")=1, py::arg("lag")=1,
        py::arg("dt")=1.e0, py::arg("k")=5, py::arg("trial")=0, py::arg("n_threads")=1,
        R"doc(Compute a transfer entropy causal map for multivariate time series.

Parameters
----------
X : ndarray of shape (N, Nt) or (N, Nt, dim)
    Multivariate time series with N variables and Nt time steps.
    Must be float64.
tau : ndarray of shape (N,), dtype int32
    Time delay for each variable.
m : int, optional
    Embedding dimension. Default 1.
lag : int, optional
    Time lag for embedding. Default 1.
dt : float, optional
    Physical time step. Default 1.0.
k : int, optional
    Number of nearest neighbors. Default 5.
trial : int, optional
    Number of surrogate trials. Default 0.
n_threads : int, optional
    Number of OpenMP threads. Default 1.

Returns
-------
value : ndarray of shape (N, N)
    Causal map where entry [i, j] is TE(X_j -> X_i).
)doc");
  m.def("information_flow_causal_map", &information_flow_causal_map_dispatcher,
        py::arg("X"),
        py::arg("tau")  = py::int_(1),
        py::arg("mask") = py::none(),
        py::arg("dt")=1.0, py::arg("k")=5, py::arg("n_threads")=1,
        R"doc(Compute an information flow causal map for multivariate time series.

Two modes depending on whether *mask* is supplied:

**No mask (default)** — full computation returning three maps:
  tau : ndarray of shape (N,), dtype int32, or scalar int
      Time delay per variable (scalar is broadcast to all variables).
  Returns : tuple of three ndarray of shape (N, N)
      (IF_map, Leak_map, dI_map).  IF_map[i, j] = IF from X_j to X_i,
      Leak_map[i, j] = associated leak, dI_map[i, j] = MI rate.
  X may be 2D (N, Nt) or 3D (N, Nt, dim).

**With mask** — compute only the requested pairs:
  tau  : scalar int.  Time delay in samples.
  mask : ndarray of shape (N, N), dtype bool.
      Compute IF only where mask[i, j] is True and i != j.
      None (default) computes all off-diagonal pairs.
  Returns : tuple of three ndarray of shape (N, N).
      (IF_map, Leak_map, dI_map).  IF_map[i, j] = IF from X_j to X_i;
      unmasked entries are 0.  Diagonal entries are NaN.
  X must be 2D (N, Nt).

Parameters common to both modes
---------------------------------
X : ndarray, float64
    Multivariate time series, one row per variable.
dt : float, optional
    Physical time step. Default 1.0.
k : int, optional
    Number of nearest neighbors. Default 5.
n_threads : int, optional
    Number of OpenMP threads. Default 1.
)doc");
}
