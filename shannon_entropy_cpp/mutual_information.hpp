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
  T eps_all = 0; 
  for (int i = 0; i < N; i++) {
    // --- XY spacekNN (Chebyshev) ---
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(0));
    T eps = out_dist[k];
    eps_all += eps;
    // --- X radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_X;
    index_X.radiusSearch(&X[i*dx], eps, matches_X, nanoflann::SearchParameters(0));
    nX[i] = matches_X.size() - 1;
    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_Y;
    index_Y.radiusSearch(&Y[i*dy], eps, matches_Y, nanoflann::SearchParameters(0));
    nY[i] = matches_Y.size() - 1;
  }
  std::cerr << "eps_mean_c++=" << eps_all / N << "\n";  // mutual information
  T digamma_sum = 0.e0;
  for (int i = 0; i < N; i++) {
    digamma_sum += - digamma(nX[i] + 1.e0) - digamma(nY[i] + 1.e0);
    }
  //Nで平均を取る
  T I = digamma(k) + digamma_sum / N + digamma(N);
  return I > 0.e0 ? I : 0.e0;
}
// ============================================================
// mutual information
// ============================================================
// epsをpyで計算したものを使用
template<typename T>
T mutual_info_noeps(T **X_ptr, T **Y_ptr, T * eps_arr, int k, int dx, int dy, int N, int Thei = 10) {
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
  T eps_all = 0; 
  for (int i = 0; i < N; i++) {
    // --- XY spacekNN (Chebyshev) ---
    //KNNResultSet<T> resultSet(k+1);
    //resultSet.init(ret_index.data(), out_dist.data());
    //index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(10));
    //T eps = out_dist[k];
    T eps = eps_arr[i];
    eps_all += eps;
    // --- X radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_X;
    index_X.radiusSearch(&X[i*dx], eps, matches_X, nanoflann::SearchParameters(0));
    nX[i] = matches_X.size() - 1;
    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_Y;
    index_Y.radiusSearch(&Y[i*dy], eps, matches_Y, nanoflann::SearchParameters(0));
    nY[i] = matches_Y.size() - 1;
  }
  std::cerr << "eps_mean_c++=" << eps_all / N << "\n";  // mutual information
  T digamma_sum = 0.e0;
  for (int i = 0; i < N; i++) {
    digamma_sum += - digamma(nX[i] + 1.e0) - digamma(nY[i] + 1.e0);
    }
  //Nで平均を取る
  T I = digamma(k) + digamma_sum / N + digamma(N);
  return I > 0.e0 ? I : 0.e0;
}

/*
// ============================================================
// mutual information
// ============================================================
#include <cmath> // std::nextafter
#include <iostream> // for std::cerr, std::cout
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
  for (int i = 0; i < N; i++) {
    // --- XY space kNN (Chebyshev) ---
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(10));
    T eps = out_dist[k];

    T eps_minus = std::nextafter(eps, std::numeric_limits<T>::lowest());
    // --- X radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_X;
    index_X.radiusSearch(&X[i*dx], eps_minus, matches_X, nanoflann::SearchParameters(10));
    nX[i] = static_cast<int>(matches_X.size()) - 1;
    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_Y;
    index_Y.radiusSearch(&Y[i*dy], eps_minus, matches_Y, nanoflann::SearchParameters(10));
    nY[i] = static_cast<int>(matches_Y.size()) - 1;

    if (i < 10) {
  std::cerr << "i="<<i<<", eps="<<eps<<", eps_minus="<<eps_minus<<"\n";
  std::cerr << "ret_idx: ";
  for (int r=0;r<=k;r++) std::cerr<< ret_index[r] << " ("<< out_dist[r] <<") ";
  std::cerr<<"\n";
  size_t knb = ret_index[k];
  // compute manual max-abs in X and Y for the k-th neighbour
  T maxdx = 0, maxdy = 0;
  for (int a=0;a<dx;a++) maxdx = std::max(maxdx, std::abs(X[i*dx + a] - X[knb*dx + a]));
  for (int a=0;a<dy;a++) maxdy = std::max(maxdy, std::abs(Y[i*dy + a] - Y[knb*dy + a]));
  std::cerr << "manual maxdx="<<maxdx<<", maxdy="<<maxdy<<", joint_max="<<std::max(maxdx,maxdy)<<"\n";
  std::cerr << "matches_X="<<matches_X.size()<<" matches_Y="<<matches_Y.size()
            << " -> nX="<<nX[i]<<" nY="<<nY[i]<<"\n";
  }
  }
  // mutual information
  T digamma_sum = 0.e0;
  for (int i = 0; i < N; i++) {
    digamma_sum += - digamma(nX[i] + 1.e0) - digamma(nY[i] + 1.e0);
    }
  //Nで平均を取る
  T I = digamma(k) + digamma_sum / N + digamma(N);
  return I > 0.e0 ? I : 0.e0;
}

*/