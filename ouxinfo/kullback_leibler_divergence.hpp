#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <cmath>
#include <vector>
#include <limits>
#include <omp.h>


using namespace nanoflann;


template <typename T>
using kd_tree_t = ChebyshevKDTree<T, PointCloud, size_t>;


// ============================================================
// Kullback-Leibler divergence
// ============================================================
template<typename T>
T KL_div(T **X_ptr, T **Y_ptr, int k, int d, int N_X, int N_Y) {
  // k-th neighbour within X (excluding the point itself) and within Y
  if (N_X <= k || N_Y < k) return std::numeric_limits<T>::quiet_NaN();
  T *X = *X_ptr;
  T *Y = *Y_ptr;

  PointCloud cloud_X, cloud_Y;
  cloud_X.N = N_X; cloud_X.dim = d; cloud_X.pts = X;
  cloud_Y.N = N_Y; cloud_Y.dim = d; cloud_Y.pts = Y;
  kd_tree_t<T> index_X(d, cloud_X, KDTreeSingleIndexAdaptorParams(10));
  kd_tree_t<T> index_Y(d, cloud_Y, KDTreeSingleIndexAdaptorParams(10));
  T   mean_log  = 0.e0;
  int valid_pts = 0;
  // indices and distances
  #pragma omp parallel reduction(+:mean_log, valid_pts)
  {
  std::vector<size_t> ret_index(k+1);
  std::vector<T> out_dist_sqr(k+1);
  
  #pragma omp for schedule(static)
  for (int i = 0; i < N_X; i++) {
    T *query_pt = &X[(size_t)i*d];
    // X-tree (k+1)
    KNNResultSet<T> resultSet_X(k+1);
    resultSet_X.init(ret_index.data(), out_dist_sqr.data());
    index_X.findNeighbors(resultSet_X, query_pt, SearchParameters(0));
    //T r = std::sqrt(out_dist_sqr[k]);
    T r = out_dist_sqr[k];
    // Y-tree (k)
    KNNResultSet<T> resultSet_Y(k);
    resultSet_Y.init(ret_index.data(), out_dist_sqr.data());
    index_Y.findNeighbors(resultSet_Y, query_pt, SearchParameters(0));
    //T s = std::sqrt(out_dist_sqr[k-1]);
    T s = out_dist_sqr[k-1];

    // as in the MI estimators, points with a zero distance (duplicates) are skipped
    if (r > 0 && s > 0) {
      mean_log += std::log(s / r);
      ++valid_pts;
    }
  }
  }
  if (valid_pts == 0) return std::numeric_limits<T>::quiet_NaN();
  mean_log /= valid_pts;
  // Kullback-Leibler divergence.  Not clipped at 0: the estimate fluctuates around the true
  // value, and clipping biases it upwards (e.g. +0.017 nats for identical distributions).
  return d * mean_log + std::log(T(N_Y)/(N_X-1));
}

