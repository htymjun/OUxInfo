#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream> // for std::cerr, std::cout


using namespace nanoflann;
using boost::math::digamma;

template <typename T>
using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
  Chebyshev_Adaptor<T, PointCloud>,
  PointCloud,
  -1,
  size_t
>;


// ============================================================
// mutual information
// ============================================================
template<typename T>
T mutual_info(T **X_ptr, T **Y_ptr, int k, int dx, int dy, int N) {
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
  for (int i = 0; i < N; i++) {
    // --- XY spacekNN (Chebyshev) ---
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(0));
    T eps = out_dist[k];
    // --- X radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_X;
    index_X.radiusSearch(&X[i*dx], eps, matches_X, nanoflann::SearchParameters(0));
    int count_X = 0;
    for (const auto& match : matches_X) {
      if (match.first != i && match.second < eps) count_X++;
    }
    nX[i] = count_X;
    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_Y;
    index_Y.radiusSearch(&Y[i*dy], eps, matches_Y, nanoflann::SearchParameters(0));
    int count_Y = 0;
    for (const auto& match : matches_Y) {
      if (match.first != i && match.second < eps) count_Y++;
    }
    nY[i] = count_Y;
  }
  // calc mutual information
  T mean_psi_nX = 0.e0;
  T mean_psi_nY = 0.e0;
  int valid_pts = 0;
  for (int i = 0; i < N; i++) {
    if (nX[i] > 0 && nY[i] > 0) {
      mean_psi_nX += digamma(nX[i] + 1.e0);
      mean_psi_nY += digamma(nY[i] + 1.e0);
      valid_pts++;
    }
  }
  T I = 0.e0;
  if (valid_pts > 0) {
    mean_psi_nX /= valid_pts;
    mean_psi_nY /= valid_pts;
    I = digamma(k) - mean_psi_nX - mean_psi_nY + digamma(N);
  }
  return I;
}


// ============================================================
// mutual information with Theiler window
// ============================================================
template<typename T>
T mutual_info_Thei(T **X_ptr, T **Y_ptr, int k, int dx, int dy, int N, int Thei) {
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
  std::vector<int> nX(N, 0), nY(N, 0), nN(N, 0);
  // Theiler window excludes kNN points at most (2*Thei+1)
  size_t num_search = std::min((size_t)N, (size_t)(k + 2 * Thei + 1));
  // indices and distances
  std::vector<size_t> ret_index(num_search);
  std::vector<T> out_dist(num_search);
  for (int i = 0; i < N; i++) {
    // --- the number of valid data ---
    int window_start = std::max(0, i - Thei);
    int window_end   = std::min(i + Thei + 1, N);
    nN[i] = N - (window_end - window_start);
    // --- XY spacekNN (Chebyshev) ---
    size_t num_results = index_XY.knnSearch(&XY[i*dxy], num_search, ret_index.data(), out_dist.data());
    // get eps out of Theiler window
    T eps = 0.e0;
    int valid_k_count = 0;
    for (size_t m = 0; m < num_results; m++) {
      int j = ret_index[m];
      if (std::abs(j - i) > Thei) {
        valid_k_count++;
        if (valid_k_count == k) {
          eps = out_dist[m];
          break;
        }
      }
    }
    // --- X radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_X;
    index_X.radiusSearch(&X[i*dx], eps, matches_X, nanoflann::SearchParameters(0));
    int count_X = 0;
    for (const auto& match : matches_X) {
      int j = match.first;
      if (std::abs(j - i) > Thei && match.second < eps) count_X++;
    }
    nX[i] = count_X;
    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_Y;
    index_Y.radiusSearch(&Y[i*dy], eps, matches_Y, nanoflann::SearchParameters(0));
    int count_Y = 0;
    for (const auto& match : matches_Y) {
      int j = match.first;
      if (std::abs(j - i) > Thei && match.second < eps) count_Y++;
    }
    nY[i] = count_Y;
  }
  // calc mutual information
  T mean_psi_nX = 0.e0;
  T mean_psi_nY = 0.e0;
  T mean_psi_nN = 0.e0;
  int valid_pts = 0;
  for (int i = 0; i < N; i++) {
    if (nX[i] > 0 && nY[i] > 0) {
      mean_psi_nX += digamma(nX[i] + 1.e0);
      mean_psi_nY += digamma(nY[i] + 1.e0);
      mean_psi_nN += digamma(nN[i] + 1.e0);
      valid_pts++;
    }
  }
  T I = 0.e0;
  if (valid_pts > 0) {
    mean_psi_nX /= valid_pts;
    mean_psi_nY /= valid_pts;
    mean_psi_nN /= valid_pts;
    I = digamma(k) - mean_psi_nX - mean_psi_nY + mean_psi_nN;
  }
  return I;
}

