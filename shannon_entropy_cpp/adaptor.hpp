#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"


// PointCloud_flat is faster than PointCloud
// PointCloud must be replaced by PointCloud_flat
typedef KDTreeSingleIndexAdaptor<
  L2_Simple_Adaptor<double, PointCloud_flat>,
  PointCloud_flat,
  -1
> kd_tree_t;

