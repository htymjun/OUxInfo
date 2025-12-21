#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "shannon_entropy.hpp"
#include "kullback_leibler_divergence.hpp"
#include "mutual_information.hpp"
#include "transfer_entropy.hpp"
#include <vector>


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
  //if (N != M) {
  //  throw std::runtime_error("Input argument must be the same length");
  //}
  int d = static_cast<int>(info_x.shape[1]);
  return KL_div(&x, &y, k, d, N, M);
}


double mutual_info_wrapper(py::array_t<double, py::array::c_style> x_obj, 
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
  if (N != M) {
    throw std::runtime_error("Input argument must be the same length");
  }
  int dx = static_cast<int>(info_x.shape[1]);
  int dy = static_cast<int>(info_y.shape[1]);
  return mutual_info(&x, &y, k, dx, dy, N);}


double transfer_entropy_wrapper(py::array_t<double, py::array::c_style> x_obj, 
                                py::array_t<double, py::array::c_style> y_obj,
                                int k=5, int tau=1, int trial=0){
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
  double TE  = transfer_entropy(&x, &y, k, dx, dy, N, tau);
  double TEs = 0.e0;
  if (trial > 0) {
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    for (int itr = 0; itr < trial; itr++) {
      std::shuffle(idx.begin(), idx.end(), g);
      std::vector<double> xs_data(N * dx);
      for (int i = 0; i < N; ++i) {
        int original_idx = idx[i];
        std::copy(x + (original_idx * dx), x + (original_idx * dx) + dx, xs_data.begin() + (i * dx));
      }
      double *xs_ptr = xs_data.data();
      TEs += transfer_entropy(&xs_ptr, &y, k, dx, dy, N, tau);
      }
  }
  return std::max(TE - TEs, 0.e0);}


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
        py::arg("X"), py::arg("Y"), py::arg("k")=5,
        "Compute mutual information of dataset X and Y using Kraskov's estimator type 1");
  m.def("transfer_entropy", &transfer_entropy_wrapper,
        py::arg("X"), py::arg("Y"), py::arg("k")=5, py::arg("tau")=1, py::arg("trial")=0,
        "Compute transfer entropy of dataset X and Y using Kraskov's estimator type 1");
}

