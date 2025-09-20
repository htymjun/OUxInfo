#pragma once
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


struct PointCloud_flat {
  size_t N;
  size_t dim;
  std::vector<double> pts;
  inline double kdtree_get_pt(const size_t idx, const size_t dim_) const {
    return pts[idx*dim + dim_];
  }
  inline size_t kdtree_get_point_count() const { return N; }
  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const { return false; }
};

