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
#include "information_flow.cpp"


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
  check_knn_k(k);
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
  check_knn_k(k);
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
  check_knn_k(k);
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
  if (Thei < 0) {
    throw std::invalid_argument("Thei must be non-negative");
  }
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
  check_knn_k(k);
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
  check_knn_k(k);
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
                                int tau=1, double dt=1.e0, int k=5, py::object z_obj=py::none()){
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
  z   : ndarray (N, dim), optional
        Conditioning variable (conditional information flow).
  Returns
  -------
  double
    information flow.
  */
  check_knn_k(k);
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
  if (tau < 0) {
    throw std::invalid_argument("tau must be non-negative");
  }
  if (z_obj.is_none()) {
    return information_flow<double>(x, y, tau, dt, k, dx, dy, N);
  }
  auto z_arr = z_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  py::buffer_info info_z = z_arr.request();
  if (info_z.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info_z.shape[0] != N) {
    throw std::runtime_error("Input argument must be the same length");
  }
  const double *z = static_cast<const double*>(info_z.ptr);
  int dz = static_cast<int>(info_z.shape[1]);
  if (dz < 1) {
    throw std::runtime_error("Z must have at least one column");
  }
  return conditional_information_flow<double>(x, y, z, tau, dt, k, dx, dy, dz, N);
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
  check_knn_k(k);
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
    Shannon entropy in nats. NaN if N <= k (no k-th neighbor).

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
    KL divergence D_KL(X || Y) in nats. NaN if N <= k or M < k.

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
    Mutual information I(X; Y) in nats. NaN if N <= k, or with a Theiler
    window if N < k + 2 * Thei + 1 (some points then have fewer than k
    admissible neighbors).

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
    Conditional mutual information I(X; Y | Z) in nats. NaN if N <= k.

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
    Transfer entropy TE(X -> Y) in nats per dt. NaN if the number of
    usable samples, N - tau - (m - 1) * lag, is at most k.

Notes
-----
Computed as conditional mutual information: TE = I(Y_{t+tau}; X_t^(m) | Y_t^(m)).
See: Schreiber (2000), Kraskov et al. (2004).
)doc");
  m.def("information_flow", &information_flow_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("tau")=1, py::arg("dt")=1.e0, py::arg("k")=5,
        py::arg("Z")=py::none(),
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
Z : ndarray of shape (N, dim_z), optional
    Conditioning time series. If given, the conditional information flow
    from X to Y given Z is returned. Default None.

Returns
-------
value : float
    Information flow dI_X (rate of mutual information change due to X) in nats per dt.
    NaN if N - tau <= k.

Notes
-----
Based on the decomposition

    dI/dt = dI_X + dI_Y + Leak

The estimate is

    T_{X->Y}   = [I(X_t; Y_{t+tau}) - I(X_t; Y_t)] / dt
    T_{X->Y|Z} = [I(X_t; Y_{t+tau} | Z_t) - I(X_t; Y_t | Z_t)] / dt

with t = 0, ..., N - tau - 1. The conditional mutual informations use the
KSG estimator of Frenzel & Pompe (2007). With Z = Y, T_{X->Y|Z} equals the
transfer entropy I(X_t; Y_{t+tau} | Y_t) / dt.

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
    Diagonal entries, and entries with at most k usable samples, are NaN.
)doc");
  m.def("information_flow_causal_map", &information_flow_causal_map_dispatcher,
        py::arg("X"),
        py::arg("tau")       = py::int_(1),
        py::arg("mask")      = py::none(),
        py::arg("dt")        = 1.0,
        py::arg("k")         = 5,
        py::arg("n_threads") = 1,
        py::arg("full")      = false,
        py::arg("Z")         = py::none(),
        R"doc(Compute an information flow causal map for multivariate time series.

Two modes depending on whether *mask* is supplied:

**No mask (default)** — full or IF-only computation:
  tau : ndarray of shape (N,), dtype int32, or scalar int
      Time delay per variable (scalar is broadcast to all variables).
  X may be 2D (N, Nt) or 3D (N, Nt, dim).

**With mask** — compute only the requested pairs:
  tau  : scalar int or ndarray of shape (N,), dtype int32.
  mask : ndarray of shape (N, N), dtype bool.
      Compute IF only where mask[i, j] is True and i != j.
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
full : bool, optional
    If False (default), return only IF_map (skips the dI MI call and Leak computation).
    If True, return a tuple (IF_map, Leak_map, dI_map).
Z : ndarray of shape (Nt,) or (Nt, dim_z), optional
    Conditioning time series shared by all pairs. If given, every mutual
    information is replaced by the conditional mutual information given Z_t
    (conditional causal map). X may then be 2D or 3D in both modes.
    Default None.

Returns
-------
When full=False : ndarray of shape (N, N)
    IF_map only.  IF_map[i, j] = IF from X_j to X_i.
    Diagonal entries are NaN.  Unmasked entries are 0 (masked mode).
When full=True : tuple of three ndarray of shape (N, N)
    (IF_map, Leak_map, dI_map).  IF_map[i, j] = IF from X_j to X_i.
    Diagonal entries are NaN.  Unmasked entries are 0 (masked mode).
Computed entries are NaN if Nt - tau <= k (no k-th neighbor).

Notes
-----
With Z, for each pair (i, j) and t = 0, ..., Nt - tau - 1,

    IF_map[i, j] = [I(X_j,t; X_i,t+tau | Z_t) - I(X_j,t; X_i,t | Z_t)] / dt
    dI_map[i, j] = [I(X_i,t+tau; X_j,t+tau | Z_t) - I(X_i,t; X_j,t | Z_t)] / dt
    Leak_map     = dI_map - IF_map - IF_map.T

Z is held at time t in every term, so dI_map is the change of I(X_i; X_j | Z_t)
caused by the evolution of X_i and X_j only, and Leak_map keeps its meaning of
the part of that change not split into the two flows. IF_map[i, j] equals the
single-pair conditional information flow from X_j to X_i (information_flow with Z).
)doc");
}
