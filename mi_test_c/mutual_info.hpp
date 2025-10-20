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
    if (N <= 0 || k <= 0) return 0.0;
    T *X = *X_ptr;
    T *Y = *Y_ptr;
    int dxy = dx + dy;

    // Joint data XY
    std::vector<T> XY(N * dxy);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < dx; j++) XY[i*dxy+j]     = X[i*dx+j];
        for (int j = 0; j < dy; j++) XY[i*dxy+dx+j] = Y[i*dy+j];
    }

    // Build KDTree
    PointCloud cloud_X, cloud_Y, cloud_XY;
    cloud_X.N   = N; cloud_X.dim   = dx;  cloud_X.pts = X;
    cloud_Y.N   = N; cloud_Y.dim   = dy;  cloud_Y.pts = Y;
    cloud_XY.N  = N; cloud_XY.dim  = dxy; cloud_XY.pts = XY.data();
    my_kd_tree_t<T> index_X(dx,   cloud_X,  KDTreeSingleIndexAdaptorParams(10));
    my_kd_tree_t<T> index_Y(dy,   cloud_Y,  KDTreeSingleIndexAdaptorParams(10));
    my_kd_tree_t<T> index_XY(dxy, cloud_XY, KDTreeSingleIndexAdaptorParams(10));
    index_X.buildIndex();
    index_Y.buildIndex();
    index_XY.buildIndex();

    std::vector<int> nX(N), nY(N);
    const T tiny_epsilon = 1e-9;

    for (int i = 0; i < N; i++) {
        std::vector<size_t> ret_index(k + 1);
        std::vector<T> out_dist(k + 1);
        KNNResultSet<T> resultSet(k + 1);
        resultSet.init(ret_index.data(), out_dist.data());
        index_XY.findNeighbors(resultSet, &XY[i * dxy], SearchParameters(10));
        T eps = out_dist[k];

        std::vector<nanoflann::ResultItem<size_t,T>> matches_X;
        index_X.radiusSearch(&X[i*dx], eps - tiny_epsilon, matches_X, SearchParameters(10));
        nX[i] = matches_X.size() - 1;

        std::vector<nanoflann::ResultItem<size_t,T>> matches_Y;
        index_Y.radiusSearch(&Y[i*dy], eps - tiny_epsilon, matches_Y, SearchParameters(10));
        nY[i] = matches_Y.size() - 1;
    }

    T digamma_sum = 0.0;
    for (int i = 0; i < N; i++) {
        digamma_sum += digamma(static_cast<T>(nX[i]) + 1.0) + digamma(static_cast<T>(nY[i]) + 1.0);
    }

    T I = digamma(static_cast<T>(k)) + digamma(static_cast<T>(N)) - digamma_sum / N;

    return I > 0.0 ? I : 0.0;
}
