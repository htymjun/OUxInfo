#ifndef K_NEAREST_NEIGHBOR
#define K_NEAREST_NEIGHBOR


std::vector<double> knn_kth_distance(const PointCloud& cloud, 
                                     const std::vector<std::vector<double>>& X_query,
                                     int k);

#endif

