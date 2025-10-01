#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include <algorithm>


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
  std::vector<size_t> ret_index(k);
  std::vector<T> out_dist(k);
  for (int i = 0; i < N; i++) {
    // --- XY spacekNN (Chebyshev) ---
    KNNResultSet<T> resultSet(k);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(10));
    T eps = out_dist[k-1];
    // --- X radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_X;
    index_X.radiusSearch(&X[i*dx], eps, matches_X, nanoflann::SearchParameters(10));
    nX[i] = matches_X.size();
    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_Y;
    index_Y.radiusSearch(&Y[i*dy], eps, matches_Y, nanoflann::SearchParameters(10));
    nY[i] = matches_Y.size();
  }
  // mutual information
  T I, digamma_nX_digamma_nY = 0.e0;
  int valid = 0;
  for (int i = 0; i < N; i++) {
    if (nX[i] > 0 && nY[i] > 0) {
      digamma_nX_digamma_nY += - digamma(nX[i] + 1.e0) - digamma(nY[i] + 1.e0);
      valid++;
    }
  }
  if (valid == 0) return 0.e0;
  I = digamma(k) + digamma_nX_digamma_nY / valid + digamma(N);
  return I > 0.e0 ? I : 0.e0;
}

