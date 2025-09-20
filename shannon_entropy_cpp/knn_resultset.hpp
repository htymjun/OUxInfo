#include "nanoflann.hpp"
#include <cmath>
#include <queue>
#include <vector>


using namespace nanoflann;


// ==========================================
// k-NN ResultSet
// ==========================================
template <typename DistanceType_>
struct kthOnlyResultSet {
  using DistanceType = DistanceType_;
  size_t k;
  std::priority_queue<DistanceType> heap;
  explicit kthOnlyResultSet(size_t k_) : k(k_) {}
  inline DistanceType worstDist() const {
    return heap.size() < k ? std::numeric_limits<DistanceType>::infinity() : heap.top();
  }
  
  inline bool full() const {return heap.size() >= k; }

  inline bool addPoint(DistanceType dist, size_t) {
    if (heap.size() < k) {
      heap.push(dist);
      return true;
    } else if (dist < heap.top()) {
      heap.pop();
      heap.push(dist);
      return true;
    }
    return false;
  }
  inline void sort() {}
};

