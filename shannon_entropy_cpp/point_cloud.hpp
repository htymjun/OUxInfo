#include "nanoflann.hpp"
#include <vector>


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

