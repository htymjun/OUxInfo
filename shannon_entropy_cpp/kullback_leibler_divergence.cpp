#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <cmath>
#include <vector>


using namespace nanoflann;


// ============================================================
// Kullback-Leibler divergence
// ============================================================
double KL_div(double **X_ptr, double **Y_ptr, int k, int d, int N) {
  if (N == 0) return 0.e0;
  double *X = *X_ptr;
  double *Y = *Y_ptr;
  PointCloud cloud_X, cloud_Y;
  cloud_X.N = N; cloud_X.dim = d; cloud_X.pts = X;
  cloud_Y.N = N; cloud_Y.dim = d; cloud_Y.pts = Y;
  kd_tree_t index_X(d, cloud_X, KDTreeSingleIndexAdaptorParams(10));
  kd_tree_t index_Y(d, cloud_Y, KDTreeSingleIndexAdaptorParams(10));
  index_X.buildIndex();
  index_Y.buildIndex();
  // indices and distances
  std::vector<size_t> ret_index(k+1);
  std::vector<double> out_dist_sqr(k+1);

  double mean_log = 0.e0;
  for (size_t i = 0; i < N; i++) {
    double *query_pt = &X[i*d];
    // X-tree (k+1)
    KNNResultSet<double> resultSet_X(k+1);
    resultSet_X.init(ret_index.data(), out_dist_sqr.data());
    index_X.findNeighbors(resultSet_X, query_pt, SearchParameters(10));
    //double r = std::sqrt(out_dist_sqr[k]);
    double r = out_dist_sqr[k];
    // Y-tree (k)
    KNNResultSet<double> resultSet_Y(k);
    resultSet_Y.init(ret_index.data(), out_dist_sqr.data());
    index_Y.findNeighbors(resultSet_Y, query_pt, SearchParameters(10));
    //double s = std::sqrt(out_dist_sqr[k-1]);
    double s = out_dist_sqr[k-1];
    mean_log += std::log(s / r);
  }
  mean_log /= N;
  // Kullback-Leibler divergence
  double Dkl = d * mean_log + std::log(double(N)/(N-1));
  return Dkl > 0.e0 ? Dkl : 0.e0;
}

