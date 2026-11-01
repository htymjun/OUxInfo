#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include <omp.h>


using namespace nanoflann;
using boost::math::digamma;


template <typename T>
using kd_tree_t = ChebyshevKDTree<T, PointCloud, size_t>;


// ============================================================
// Shannon entropy
// ============================================================
template <typename T>
T shannon_entropy(T **X_ptr, int k, int d, int N) {
  if (N == 0) return 0.e0;
  // KDTree
  T *X = *X_ptr;
  PointCloud cloud;
  cloud.N = N;
  cloud.dim = d;
  cloud.pts = X;

  kd_tree_t<T> index(d, cloud, KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();

  T mean_log_eps = 0.e0;

  #pragma omp parallel
  {
  // indices and distances
  std::vector<size_t> ret_index(k+1);
  std::vector<T> out_dist_sqr(k+1);
  T local_sum = 0.e0;

  // epsilong and E(log(epsilon))
  #pragma omp for nowait
  for (int i = 0; i < N; i++) {
    T *query_pt = &X[i*d];
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist_sqr.data());
    index.findNeighbors(resultSet, query_pt, SearchParameters(0));
    //eps = 2.e0 * std::sqrt(out_dist_sqr[k]);
    T eps = 2.e0 * out_dist_sqr[k];
    //mean_log_eps += std::log(eps);
    local_sum += std::log(eps);
  }
  #pragma omp atomic
    mean_log_eps += local_sum;
}
  mean_log_eps /= N;
  // eps is the diameter of the max-norm ball, whose volume is eps^d (c_d = 1).
  // The Euclidean constant pi^{d/2}/Gamma(1+d/2)/2^d would be wrong here: the
  // distances come from the Chebyshev tree (it coincides with 1 only for d = 1).
  // Shannon entropy
  T H = - digamma(k) + digamma(N) + d * mean_log_eps;
  return H;
}
