#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fortran_interface.h"


namespace py = pybind11;


float shannon_entropy_wrapper(py::array_t<float, py::array::f_style | py::array::c_style> x_obj, int k=5){
  py::buffer_info info = x_obj.request();
  if (info.ndim != 2) {
    throw std::runtime_error("Input dimension must be 2");
  }
  if (info.itemsize != sizeof(float)) {
    throw std::runtime_error("Expected float32");
  }
  float *x = static_cast<float*>(info.ptr);
  ssize_t rows = info.shape[0];
  ssize_t cols = info.shape[1];
  int N  = static_cast<int>(rows);
  int dx = static_cast<int>(cols);
  
  void *cp = (void*)x;
  float H;
  shannon_entropy_cbind(k, dx, N, &cp, &H);
  return H;
}


PYBIND11_MODULE(shannon, m) {
  m.def("shannon_entropy", &shannon_entropy_wrapper,
        py::arg("x"), py::arg("k"));
}

