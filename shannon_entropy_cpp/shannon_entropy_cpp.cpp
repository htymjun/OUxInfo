#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "shannon_entropy.hpp"
#include "kullback_leibler_divergence.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>


namespace py = pybind11;
using namespace nanoflann;


// ============================================================
// pybind11 module
// ============================================================
PYBIND11_MODULE(shannon_entropy_cpp, m) {
  m.doc() = "Shannon entropy using nanoflann + Boost digamma";
  m.def("shannon_entropy", &shannon_entropy,
        py::arg("X"), py::arg("k")=3,
        "Compute Shannon entropy of dataset X using Kozachenko-Leonenko estimator");
  m.def("KL_div", &KL_div,
        py::arg("X"), py::arg("Y"), py::arg("k")=3,
        "Compute Kullback-Leibler divergence of dataset X and Y using Pérez-Cruz");
}

