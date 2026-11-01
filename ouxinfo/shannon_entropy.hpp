#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include <limits>
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
  if (N <= k) return std::numeric_limits<T>::quiet_NaN();
  // KDTree
  T *X = *X_ptr;
  PointCloud cloud;
  cloud.N = N;
  cloud.dim = d;
  cloud.pts = X;

  kd_tree_t<T> index(d, cloud, KDTreeSingleIndexAdaptorParams(10));

  T   mean_log_eps = 0.e0;
  int valid_pts    = 0;

  // reduction as in the MI estimators (the atomic update made the sum order depend on which
  // thread finished first)
  #pragma omp parallel reduction(+:mean_log_eps, valid_pts)
  {
  // indices and distances
  std::vector<size_t> ret_index(k+1);
  std::vector<T> out_dist_sqr(k+1);

  // epsilong and E(log(epsilon))
  #pragma omp for schedule(static)
  for (int i = 0; i < N; i++) {
    T *query_pt = &X[(size_t)i*d];
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist_sqr.data());
    index.findNeighbors(resultSet, query_pt, SearchParameters(0));
    T eps = 2.e0 * out_dist_sqr[k];
    // as in the MI estimators, points with eps == 0 (k or more duplicates) are skipped
    if (eps > 0) {
      mean_log_eps += std::log(eps);
      ++valid_pts;
    }
  }
  }
  if (valid_pts == 0) return std::numeric_limits<T>::quiet_NaN();
  mean_log_eps /= valid_pts;
  // eps is the diameter of the max-norm ball, whose volume is eps^d (c_d = 1).
  // The Euclidean constant pi^{d/2}/Gamma(1+d/2)/2^d would be wrong here: the
  // distances come from the Chebyshev tree (it coincides with 1 only for d = 1).
  // Shannon entropy
  T H = - digamma(k) + digamma(N) + d * mean_log_eps;
  return H;
}
