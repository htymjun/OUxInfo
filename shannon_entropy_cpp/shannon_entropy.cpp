#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include "shannon_entropy.hpp"


namespace py = pybind11;
using namespace nanoflann;


// ============================================================
// Shannon entropy
// ============================================================
double shannon_entropy(double **X_ptr, int k, int d, int N) {
  if (N == 0) return 0.e0;
  // KDTree
  double *X = *X_ptr;
  PointCloud_flat cloud;
  cloud.N   = N;
  cloud.dim = d;
  cloud.pts = X;
  kd_tree_t index(d, cloud, KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();
  // indices and distances
  std::vector<size_t> ret_index(k+1);
  std::vector<double> out_dist_sqr(k+1);
  // epsilong and E(log(epsilon))
  double eps, mean_log_eps = 0.e0;
  for (size_t i = 0; i < N; i++) {
    double *query_pt = &X[i*d];
    KNNResultSet<double> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist_sqr.data());
    index.findNeighbors(resultSet, query_pt, SearchParameters(10));
    eps = 2.e0 * std::sqrt(out_dist_sqr[k]);
    mean_log_eps += std::log(eps);
  }
  mean_log_eps /= N;
  // volume of unit ball C_d
  double pi = acos(-1.e0);
  double Cd = std::pow(pi, 0.5e0 * d) / std::tgamma(1.e0 + 0.5e0 * d) / std::pow(2.e0, d);
  // Shannon entropy
  double H = - boost::math::digamma(double(k)) + boost::math::digamma(double(N)) 
              + std::log(Cd) + d * mean_log_eps;
  return H;
}

