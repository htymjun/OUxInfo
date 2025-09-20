#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "shannon_entropy.hpp"
#include "kullback_leibler_divergence.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>


namespace py = pybind11;
using namespace nanoflann;


double shannon_entropy_wrapper(py::array_t<double, py::array::c_style> x_obj, int k=5){
  py::buffer_info info = x_obj.request();
  if (info.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info.itemsize != sizeof(double)) {
    throw std::runtime_error("Expected float64");
  }
  double *x = static_cast<double*>(info.ptr);
  ssize_t rows = info.shape[0];
  ssize_t cols = info.shape[1];
  int N = static_cast<int>(rows);
  int d = static_cast<int>(cols);
  
  return shannon_entropy(&x, k, d, N);
}

/*
double KL_div_wrapper(py::array_t<double, py::array::c_style> x_obj, 
                      py::array_t<double, py::array::c_style> y_obj, int k=5){
  py::buffer_info info_x = x_obj.request();
  py::buffer_info info_y = y_obj.request();
  if (info_x.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info_x.itemsize != sizeof(double)) {
    throw std::runtime_error("Expected float64");
  }
  double *x = static_cast<double*>(info_x.ptr);
  double *y = static_cast<double*>(info_y.ptr);
  ssize_t rows_x = info_x.shape[0];
  ssize_t rows_y = info_y.shape[0];
  ssize_t cols_x = info_x.shape[1];
  int N = static_cast<int>(rows_x);
  int M = static_cast<int>(rows_y);
  int d = static_cast<int>(cols_x);
  
  return KL_div(&x, &y, k, d, N, M);
}
*/

// ============================================================
// pybind11 module
// ============================================================
PYBIND11_MODULE(shannon_entropy_cpp, m) {
  m.doc() = "Shannon entropy using nanoflann + Boost digamma";
  m.def("shannon_entropy", &shannon_entropy_wrapper,
        py::arg("X"), py::arg("k")=3,
        "Compute Shannon entropy of dataset X using Kozachenko-Leonenko estimator");
  m.def("KL_div", &KL_div,
        py::arg("X"), py::arg("Y"), py::arg("k")=3,
        "Compute Kullback-Leibler divergence of dataset X and Y using Pérez-Cruz");
}

