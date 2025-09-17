#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nanoflann.hpp"
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using namespace nanoflann;

// =====================================
// 点群データのラッパー
// =====================================
struct PointCloud {
    std::vector<std::vector<double>> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx][dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// =====================================
// k 番目近傍点探索関数
// =====================================
py::array_t<double> kth_neighbor_distance(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    int k
) 
{
    if (X.ndim() != 2)
        throw std::runtime_error("X must be 2-dimensional");

    size_t N = X.shape(0);
    size_t d = X.shape(1);

    // PointCloud にコピー
    PointCloud cloud;
    cloud.pts.resize(N);
    auto X_unchecked = X.unchecked<2>();
    for (size_t i = 0; i < N; i++) {
        cloud.pts[i].resize(d);
        for (size_t j = 0; j < d; j++)
            cloud.pts[i][j] = X_unchecked(i,j);
    }

    // nanoflann KDTree
    typedef KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<double, PointCloud>,
        PointCloud,
        -1
    > kd_tree_t;

    kd_tree_t index(d, cloud, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    // 結果用配列 (各点に対して k 番目の距離のみ)
    py::array_t<double> kth_dists({N});
    auto kd = kth_dists.mutable_unchecked<1>();

    std::vector<size_t> ret_index(k);
    std::vector<double> out_dist_sqr(k);
    double *query_pt = new double[d];

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < d; j++) query_pt[j] = cloud.pts[i][j];

        KNNResultSet<double> resultSet(k);
        resultSet.init(ret_index.data(), out_dist_sqr.data());
        index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters(10));

        kd(i) = out_dist_sqr[k-1];  // k番目の距離だけ返す
    }

    delete[] query_pt;
    return kth_dists;
}

// =====================================
// pybind11 モジュール定義
// =====================================
PYBIND11_MODULE(nanoflann_knn, m) {
    m.doc() = "nanoflann k-th nearest neighbor distances via pybind11";
    m.def("kth_neighbor_distance", &kth_neighbor_distance, py::arg("X"), py::arg("k"));
}

