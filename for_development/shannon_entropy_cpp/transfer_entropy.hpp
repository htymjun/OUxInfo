#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
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
// transfer entropy
// ============================================================
template<typename T>
T transfer_entropy(T **x_ptr, T **y_ptr, int k, int dx, int dy, int Nt, int tau = 1) {
  // Transfer entropy from x to y
  if (Nt == 0) return 0.e0;
  T *x = *x_ptr;
  T *y = *y_ptr;
  int N = Nt - tau;
  int dz = dy;
  int dxyz = dx + dy + dz; int dyz = dy + dz; int dxz = dx + dz;
  std::vector<T> X(N * dx), Y(N * dy), Z(N * dz);
  for (int i = 0; i < N; i++) {
    X[i] = x[i]; Y[i] = y[i+tau]; Z[i] = y[i];
  }
  // Joint data XYZ, Z, YZ, XZ
  std::vector<T> XYZ(N * dxyz), YZ(N * dyz), XZ(N * dxz);
  for (int i = 0; i < N; i++) {
    // XYZ = XYZ(X, Y, Z)
    for (int j = 0; j < dx; j++) XYZ[i*dxyz+j]       = X[i*dx+j];
    for (int j = 0; j < dy; j++) XYZ[i*dxyz+dx+j]    = Y[i*dy+j];
    for (int j = 0; j < dz; j++) XYZ[i*dxyz+dx+dy+j] = Z[i*dz+j];
    // YZ = YZ(Y, Z)
    for (int j = 0; j < dy; j++) YZ[i*dyz+j]         = Y[i*dy+j];
    for (int j = 0; j < dz; j++) YZ[i*dyz+dy+j]      = Z[i*dz+j];
    // XZ = XZ(X, Z)
    for (int j = 0; j < dx; j++) XZ[i*dxz+j]         = X[i*dx+j];
    for (int j = 0; j < dz; j++) XZ[i*dxz+dx+j]      = Z[i*dz+j];
  }
  // Build KDTree
  PointCloud cloud_XYZ, cloud_Z, cloud_YZ, cloud_XZ;
  cloud_XYZ.N = N; cloud_XYZ.dim = dxyz; cloud_XYZ.pts = XYZ.data();
  cloud_Z.N   = N; cloud_Z.dim   = dz;   cloud_Z.pts   = Z.data();
  cloud_YZ.N  = N; cloud_YZ.dim  = dyz;  cloud_YZ.pts  = YZ.data();
  cloud_XZ.N  = N; cloud_XZ.dim  = dxz;  cloud_XZ.pts  = XZ.data();
  my_kd_tree_t<T> index_XYZ(dxyz, cloud_XYZ, KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t<T> index_Z  (dz,   cloud_Z,   KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t<T> index_YZ (dyz,  cloud_YZ,  KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t<T> index_XZ (dxz,  cloud_XZ,  KDTreeSingleIndexAdaptorParams(10));
  index_XYZ.buildIndex();
  index_Z.buildIndex();
  index_YZ.buildIndex();
  index_XZ.buildIndex();
  std::vector<int> nZ(N, 0), nYZ(N, 0), nXZ(N, 0);
  // indices and distances
  std::vector<size_t> ret_index(k+1);
  std::vector<T> out_dist(k+1);
  //T eps_all = 0; 
  for (int i = 0; i < N; i++) {
    // --- XY spacekNN (Chebyshev) ---
    KNNResultSet<T> resultSet(k+1);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XYZ.findNeighbors(resultSet, &XYZ[i * dxyz], SearchParameters(0));
    T eps = out_dist[k];
    //eps_all += eps;
    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_Z;
    index_Z.radiusSearch(&Z[i*dz],    eps, matches_Z,  nanoflann::SearchParameters(0));
    nZ[i] = matches_Z.size() - 1;
    // --- YZ radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_YZ;
    index_YZ.radiusSearch(&YZ[i*dyz], eps, matches_YZ, nanoflann::SearchParameters(0));
    nYZ[i] = matches_YZ.size() - 1;
    // --- XZ radius search ---
    std::vector<nanoflann::ResultItem<size_t,T>> matches_XZ;
    index_XZ.radiusSearch(&XZ[i*dxz], eps, matches_XZ, nanoflann::SearchParameters(0));
    nXZ[i] = matches_XZ.size() - 1;
  }
  //std::cerr << "eps_mean_c++=" << eps_all / N << "\n";  // transfer entropy
  T digamma_sum = 0.e0;
  for (int i = 0; i < N; i++) {
    digamma_sum += digamma(nZ[i] + 1.e0) - digamma(nXZ[i] + 1.e0) - digamma(nYZ[i] + 1.e0);
  }
  T TE = digamma(k) + digamma_sum / N;
  return TE > 0.e0 ? TE : 0.e0;
}

