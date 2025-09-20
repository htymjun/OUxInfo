#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "knn_resultset.hpp"
#include <cmath>
#include <queue>
#include <vector>
#include "k_nearest_neighbor.hpp"


using namespace nanoflann;


// ==========================================
// k-NN distance
// ==========================================
std::vector<double> knn_kth_distance(const PointCloud& cloud, 
                                     const std::vector<std::vector<double>>& X_query,
                                     int k)
{
  size_t N = X_query.size();
  size_t d = X_query[0].size();
  typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    -1
  > kd_tree_t;
  kd_tree_t index(d, cloud, KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();
  std::vector<double> kth_dist(N);
  std::vector<size_t> ret_index(k);
  std::vector<double> out_dist_sqr(k);
  double* query_pt = new double[d];
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < d; j++) query_pt[j] = X_query[i][j];
      KNNResultSet<double> resultSet(k);
      resultSet.init(ret_index.data(), out_dist_sqr.data());
      index.findNeighbors(resultSet, query_pt, SearchParameters(10));
      kth_dist[i] = std::sqrt(out_dist_sqr[k-1]);
  }
  delete[] query_pt;
  return kth_dist;
}


/*
// ==========================================
// k-NN distance
// ==========================================
std::vector<double> knn_kth_distance(const PointCloud& cloud, 
                                     const std::vector<std::vector<double>>& X_query,
                                     int k)
{
  size_t N = X_query.size();
  size_t d = X_query[0].size();
  typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    -1
  > kd_tree_t;
  kd_tree_t index(d, cloud, KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();
  std::vector<double> kth_dist(N);
  std::vector<double> query_pt(d);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < d; j++) query_pt[j] = X_query[i][j];
      kthOnlyResultSet<double> resultSet(k);
      index.findNeighbors(resultSet, query_pt.data(), SearchParameters(10));
      kth_dist[i] = std::sqrt(resultSet.worstDist());
  }
  return kth_dist;
}
*/

