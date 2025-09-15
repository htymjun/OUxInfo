#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fortran_interface.h"


namespace py = pybind11;


void shannon_entropy_wrapper(int k, int dx, int N, py::array_t<float> x_obj){
  py::buffer_info info = x_obj.request();
  float *x = static_cast<float*>(info.ptr);
  shannon_entropy_cbind(k, dx, N, x);
}


PYBIND11_MODULE(shannon, m) {
  m.def("shannon_entropy", &shannon_entropy_wrapper,
        py::arg("k"), py::arg("dx"), py::arg("N"), py::arg("x"));
}

