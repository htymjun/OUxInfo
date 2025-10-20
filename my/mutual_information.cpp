#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include "adaptor.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <cmath>
#include <vector>
#include <algorithm>


using namespace nanoflann;
using boost::math::digamma;
using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
  Chebyshev_Adaptor<double, PointCloud>,
  PointCloud,
  -1,
  size_t
>;

// ============================================================
// mutual information
// ============================================================
double mutual_info_wrapper(double **X_ptr, double **Y_ptr, int k, int dx, int dy, int N, int Thei = 10) {
  if (N == 0) return 0.e0;
  double *X = *X_ptr;
  double *Y = *Y_ptr;
  int dxy = dx + dy;
  // Joint data XY
  std::vector<double> XY(N * dxy);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < dx; j++) XY[i*dxy+j]    = X[i*dx+j];
    for (int j = 0; j < dy; j++) XY[i*dxy+dx+j] = Y[i*dy+j];
  }
  // Build KDTree
  PointCloud cloud_X, cloud_Y, cloud_XY;
  cloud_X.N  = N; cloud_X.dim  = dx;  cloud_X.pts = X;
  cloud_Y.N  = N; cloud_Y.dim  = dy;  cloud_Y.pts = Y;
  cloud_XY.N = N; cloud_XY.dim = dxy; cloud_XY.pts = XY.data();
  my_kd_tree_t index_X(dx,   cloud_X,  KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t index_Y(dy,   cloud_Y,  KDTreeSingleIndexAdaptorParams(10));
  my_kd_tree_t index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));
  index_X.buildIndex();
  index_Y.buildIndex();
  index_XY.buildIndex();
  std::vector<double> nX(N, 0.e0), nY(N, 0.e0), nN(N, 0.e0);
  // indices and distances
  std::vector<size_t> ret_index(k);
  std::vector<double> out_dist(k);

  for (int i = 0; i < N; i++) {
    // --- Theiler window ---
    std::vector<bool> valid_mask(N, true);
    int start = std::max(0, i - Thei);
    int end   = std::min(N, i + Thei + 1);
    for (int j = start; j < end; j++) valid_mask[j] = false;

    // --- XY spacekNN (Chebyshev) ---
    KNNResultSet<double> resultSet(k);
    resultSet.init(ret_index.data(), out_dist.data());
    index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(10));
    double eps = out_dist[k-1];

    // --- X radius search ---
    std::vector<nanoflann::ResultItem<size_t,double>> matches_X;
    index_X.radiusSearch(&X[i*dx], eps, matches_X, nanoflann::SearchParameters(10));
    int countX = 0;
    for (auto &m : matches_X) {
      if (valid_mask[m.first]) countX++;
    }

    // --- Y radius search ---
    std::vector<nanoflann::ResultItem<size_t,double>> matches_Y;
    index_Y.radiusSearch(&Y[i*dy], eps, matches_Y, nanoflann::SearchParameters(10));
    int countY = 0;
    for (auto &m : matches_Y) {
      if (valid_mask[m.first]) countY++;
    }

    // --- (Theiler window) ---
    int countN = 0;
    for (int j = 0; j < N; j++) if (valid_mask[j]) countN++;
    nX[i] = countX;
    nY[i] = countY;
    nN[i] = countN; 
  }
  // mutual information
  double I = 0.e0;
  int valid = 0;
  for (int i = 0; i < N; i++) {
    if (nX[i] > 0 && nY[i] > 0) {
      I += digamma(k) - digamma(nX[i] + 1.e0) - digamma(nY[i] + 1.e0) + digamma(nN[i] + 1.e0);
      valid++;
    }
  }
  if (valid == 0) return 0.e0;
  I /= valid;
  return I > 0.e0 ? I : 0.e0;
}
