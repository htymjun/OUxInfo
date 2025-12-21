#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream> // for std::cerr, std::cout
#include <omp.h>


using namespace nanoflann;
using boost::math::digamma;
template <typename T>
using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
  //L2_Simple_Adaptor<T, PointCloud>,
  Chebyshev_Adaptor<T, PointCloud>,
  PointCloud,
  -1,
  size_t
>;


// ============================================================
// mutual information
// ============================================================
template<typename T>
T mutual_info(T **X_ptr, T **Y_ptr, int k, int dx, int dy, int N, int Thei = 10) {
  if (N == 0) return 0.e0;
  T *X = *X_ptr;
  T *Y = *Y_ptr;
  int dxy = dx + dy;
  // Joint data XY
  std::vector<T> XY(N * dxy);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XY[i*dxy+j]    = X[i*dx+j];
    for (int j = 0; j < dy; j++) XY[i*dxy+dx+j] = Y[i*dy+j];
  }
  // Build KDTree
  PointCloud cloud_X, cloud_Y, cloud_XY;
  cloud_X.N  = N; cloud_X.dim  = dx;  cloud_X.pts = X;
  cloud_Y.N  = N; cloud_Y.dim  = dy;  cloud_Y.pts = Y;
  cloud_XY.N = N; cloud_XY.dim = dxy; cloud_XY.pts = XY.data();
  my_kd_tree_t<T> index_X(dx,   cloud_X,  KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t<T> index_Y(dy,   cloud_Y,  KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t<T> index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));
  index_X.buildIndex();
  index_Y.buildIndex();
  index_XY.buildIndex();
  std::vector<int> nX(N, 0), nY(N, 0);
  // indices and distances
  std::vector<size_t> ret_index(k+1);
  std::vector<T> out_dist(k+1);
  //T eps_all = 0; 
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    // --- XY spacekNN (Chebyshev) ---
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(0));
    T eps = out_dist[k];
    //eps_all += eps;
    // --- X radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_X;
    index_X.radiusSearch(&X[i*dx], eps, matches_X, nanoflann::SearchParameters(0));
    nX[i] = matches_X.size() - 1;
    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_Y;
    index_Y.radiusSearch(&Y[i*dy], eps, matches_Y, nanoflann::SearchParameters(0));
    nY[i] = matches_Y.size() - 1;
  }
  //std::cerr << "eps_mean_c++=" << eps_all / N << "\n";  // mutual information
  T digamma_sum = 0.e0;
  #pragma omp parallel for reduction(+:digamma_sum)
  for (int i = 0; i < N; i++) {
    digamma_sum += - digamma(nX[i] + 1.e0) - digamma(nY[i] + 1.e0);
    }
  //Nで平均を取る
  T I = digamma(k) + digamma_sum / N + digamma(N);
  return I > 0.e0 ? I : 0.e0;
}

