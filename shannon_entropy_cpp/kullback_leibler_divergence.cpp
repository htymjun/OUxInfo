#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include "k_nearest_neighbor.hpp"
#include "kullback_leibler_divergence.hpp"


namespace py = pybind11;
using namespace nanoflann;


// ============================================================
// Kullback-Leibler divergence
// ============================================================
double KL_div(const std::vector<std::vector<double>>& X,
              const std::vector<std::vector<double>>& Y,
              int k) {
  size_t N = X.size();
  size_t M = Y.size();
  if (N == 0 || M == 0) return 0.e0;
  size_t d = X[0].size();
  // KDTree
  PointCloud cloud_X{X};
  PointCloud cloud_Y{Y};
  std::vector<double> r = knn_kth_distance(cloud_X, X, k+1);
  std::vector<double> s = knn_kth_distance(cloud_Y, Y, k);
  // Kullback-Leibler divergence
  double mean_log = 0.e0;
  for (size_t i = 0; i < N; i++) mean_log += std::log(s[i] / r[i]);
  mean_log /= N;
  double Dkl = d * mean_log + std::log(double(M)/(N-1));
  return Dkl > 0.e0 ? Dkl : 0.e0;
}

