#include "nanoflann.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace nanoflann;

// ============================================================
// 点群を保持するシンプルな構造体
// ============================================================
struct PointCloud {
  struct Point {
    double x, y;
  };
  std::vector<Point> pts;

  // 必須: データ数を返す
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // 必須: dim=0ならx, dim=1ならyを返す
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return (dim == 0 ? pts[idx].x : pts[idx].y);
  }

  // オプション: バウンディングボックス (不要ならfalseを返せばOK)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const { return false; }
};

int main() {
  // -----------------------------------------
  // 1. データ生成
  // -----------------------------------------
  PointCloud cloud;
  for (size_t i = 0; i < 10; i++) {
    cloud.pts.push_back({double(rand() % 100), double(rand() % 100)});
  }

  // -----------------------------------------
  // 2. kd-tree 構築
  // -----------------------------------------
  typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    2 /* 次元 */
  > my_kd_tree_t;

  my_kd_tree_t index(2 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /*max leaf*/));
  index.buildIndex();

  // -----------------------------------------
  // 3. 最近傍探索
  // -----------------------------------------
  double query_pt[2] = {50.0, 50.0};
  size_t num_results = 1;
  std::vector<size_t> ret_index(num_results);
  std::vector<double> out_dist_sqr(num_results);

  KNNResultSet<double> resultSet(num_results);
  resultSet.init(&ret_index[0], &out_dist_sqr[0]);
  index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));

  // -----------------------------------------
  // 4. 結果表示
  // -----------------------------------------
  std::cout << "Query point: (" << query_pt[0] << ", " << query_pt[1] << ")\n";
  std::cout << "Nearest point index = " << ret_index[0]
            << "  coords=(" << cloud.pts[ret_index[0]].x
            << ", " << cloud.pts[ret_index[0]].y << ")"
            << "  dist^2=" << out_dist_sqr[0] << std::endl;

  return 0;
}

