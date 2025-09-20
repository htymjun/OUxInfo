#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"


typedef KDTreeSingleIndexAdaptor<
  L2_Simple_Adaptor<double, PointCloud>,
  PointCloud,
  -1
> kd_tree_t;

