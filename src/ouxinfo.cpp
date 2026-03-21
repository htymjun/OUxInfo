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


namespace py = pybind11;


double shannon_entropy_wrapper(py::array_t<double, py::array::c_style> x_obj, int k=5){
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
  return std::max(TE - TEs, 0.e0);
}


double information_flow_wrapper(py::array_t<double, py::array::c_style> x_obj, 
                                py::array_t<double, py::array::c_style> y_obj,
                                int tau=1, double dt=1.e0, int k=5){
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
  // shift data
  int Neff = N - tau;
  double *z = nullptr;
  z = y + tau * dy;
  double Ilag = mutual_info(&x, &z, k, dx, dy, Neff);
  double I    = mutual_info(&x, &y, k, dx, dy, Neff);
  return (Ilag - I) / dt;
}


py::array_t<double> transfer_entropy_causal_map_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::array_t<int,    py::array::c_style> tau_obj,
  int m=1, int lag=1, double dt=1.e0, int k=5, int trial=0, int n_threads=1){
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
        TE[j*N + i] = std::max(TE_val - TEs, 0.0) / dt;
      }
    }
  }
  return TE_map;
}


py::array_t<double> information_flow_causal_map_wrapper(
  py::array_t<double, py::array::c_style> X_obj,
  py::array_t<int,    py::array::c_style> tau_obj,
  double dt=1.e0, int k=5, int n_threads=1){
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
  py::array_t<double> IF_map({N,N});
  py::buffer_info info_out = IF_map.request();
  double *IF = static_cast<double*>(info_out.ptr);
  // main loop
  omp_set_num_threads(n_threads);
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic)
    for (int j = 0; j < N; j++) {
      double *xj = X + j * Nt * dx;
      for (int i = 0; i < N; i++) {
        if (i == j) {
          IF[j*N + i] = std::numeric_limits<double>::quiet_NaN();
          continue;
        }
        int tau = tau_arr[i];
        double *xi = X + i * Nt * dx;
        /// shift
        int Neff = Nt - tau;
        double *z  = nullptr;
        z = xj + tau * dx;
        double Ilag = mutual_info(&xi, &z,  k, dx, dx, Neff);
        double I    = mutual_info(&xi, &xj, k, dx, dx, Neff);
        IF[j*N + i] = (Ilag - I) / dt;
      }
    }
  }
  return IF_map;
}


// ============================================================
// pybind11 module
// ============================================================
PYBIND11_MODULE(ouxinfo, m) {
  m.doc() = "Shannon entropy using nanoflann + Boost digamma";
  m.def("shannon_entropy", &shannon_entropy_wrapper,
        py::arg("X"), py::arg("k")=5,
        "Compute Shannon entropy of dataset X using Kozachenko-Leonenko estimator");
  m.def("KL_div", &KL_div_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("k")=5,
        "Compute Kullback-Leibler divergence of dataset X and Y using Pérez-Cruz");
  m.def("mutual_info", &mutual_info_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("k")=5, py::arg("Thei")=0,
        "Compute mutual information of dataset X and Y using Kraskov's estimator type 1");
  m.def("conditional_mutual_info", &conditional_mutual_info_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("Z"), py::arg("k")=5,
        "Compute conditional mutual information of dataset X, Y, and Z using Kraskov's estimator type 1");
  m.def("transfer_entropy", &transfer_entropy_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("tau")=1, py::arg("m")=1, py::arg("lag")=1,
        py::arg("dt")=1.e0, py::arg("k")=5, py::arg("trial")=0,
        "Compute transfer entropy of dataset X and Y using Kraskov's estimator type 1");
  m.def("information_flow", &information_flow_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("tau")=1, py::arg("dt")=1.e0, py::arg("k")=5,
        "Compute information flow of dataset X and Y using Kraskov's estimator type 1");
  m.def("transfer_entropy_causal_map", &transfer_entropy_causal_map_wrapper,
        py::arg("X"), py::arg("tau"), py::arg("m")=1, py::arg("lag")=1,
        py::arg("dt")=1.e0, py::arg("k")=5, py::arg("trial")=0, py::arg("n_threads")=1,
        "Compute causal map based on transfer entropy");
  m.def("information_flow_causal_map", &information_flow_causal_map_wrapper,
        py::arg("X"), py::arg("tau"), py::arg("dt")=1.e0, py::arg("k")=5, py::arg("n_threads")=1,
        "Compute causal map based on information flow");
}

