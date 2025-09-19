#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nanoflann.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>


namespace py = pybind11;
using namespace nanoflann;


// ============================================================
// point data wrapper
// ============================================================
struct PointCloud {
  std::vector<std::vector<double>> pts;
  inline size_t kdtree_get_point_count() const { return pts.size(); }
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return pts[idx][dim];
  }
  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const { return false; }
};


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


// ============================================================
// Shannon entropy
// ============================================================
double shannon_entropy(const std::vector<std::vector<double>>& X,
                       int k = 3) {
  size_t N = X.size();
  if (N == 0) return 0.e0;
  size_t d = X[0].size();
  // KDTree
  PointCloud cloud;
  cloud.pts = X;
  typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    -1
  > kd_tree_t;
  kd_tree_t index(d, cloud, KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();
  // indices and distances
  std::vector<size_t> ret_index(k+1);
  std::vector<double> out_dist_sqr(k+1);
  double* query_pt = new double[d];
  // epsilong and E(log(epsilon))
  double eps, mean_log_eps = 0.e0;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < d; j++) query_pt[j] = X[i][j];
    KNNResultSet<double> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist_sqr.data());
    index.findNeighbors(resultSet, query_pt, SearchParameters(10));
    eps = 2.e0 * std::sqrt(out_dist_sqr[k]);
    mean_log_eps += std::log(eps);
  }
  mean_log_eps /= N;
  delete[] query_pt;
  // volume of unit ball C_d
  double pi = acos(-1.e0);
  double Cd = std::pow(pi, 0.5e0 * d) / std::tgamma(1.e0 + 0.5e0 * d) / std::pow(2.e0, d);
  // Shannon entropy
  double H = - boost::math::digamma(double(k)) + boost::math::digamma(double(N)) 
              + std::log(Cd) + d * mean_log_eps;
  return H;
}


// ============================================================
// Kullback-Leibler divergence
// ============================================================
double KL_div(const std::vector<std::vector<double>>& X,
              const std::vector<std::vector<double>>& Y,
              int k = 3) {
  size_t N = X.size();
  size_t M = Y.size();
  if (N == 0 || M == 0) return 0.e0;
  size_t d = X[0].size();
  // KDTree
  PointCloud cloud_X{X};
  PointCloud cloud_Y{Y};
  std::vector<double> r = knn_kth_distance(cloud_X, X, k+1);
  std::vector<double> s = knn_kth_distance(cloud_Y, X, k);
  // Kullback-Leibler divergence
  double mean_log = 0.e0;
  for (size_t i = 0; i < N; i++) mean_log += std::log(s[i] / r[i]);
  mean_log /= N;
  double Dkl = d * mean_log + std::log(double(M)/(N-1));
  return Dkl > 0.e0 ? Dkl : 0.e0;
}


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

