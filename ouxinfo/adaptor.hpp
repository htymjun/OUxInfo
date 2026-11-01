#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include <type_traits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>


using namespace nanoflann;


// --- Chebyshev (L-infty) distance adaptor for nanoflann ---
// Compatible with nanoflann internals: provides ElementType, DistanceType, ResultType,
// evalMetric(const ElementType*, const ElementType*, size_t),
// operator()(const ElementType*, size_t, size_t, ResultType),
// and accum_dist overloads that handle both scalar and array-like arguments.
template <class T, class DataSource>
struct Chebyshev_Adaptor {
  typedef T ElementType;
  typedef T ResultType;
  typedef T DistanceType;

  const DataSource& data_source;

  Chebyshev_Adaptor(const DataSource& _data_source) : data_source(_data_source) {}

  // --- evalMetric: pointer <-> pointer ---
  inline ResultType evalMetric(const ElementType *a, const ElementType *b, size_t size) const
  {
    ResultType dist = ResultType(0);
    for (size_t i = 0; i < size; i++) {
      ResultType diff = std::abs(a[i] - b[i]);
      if (diff > dist) dist = diff;
    }
    return dist;
  }

  // --- evalMetric: pointer <-> index (dataset index) ---
  inline ResultType evalMetric(const ElementType *a, const size_t b_idx, size_t size) const
  {
    return operator()(a, b_idx, size);
  }

  // --- operator() : query-pointer and dataset index ---
  inline ResultType operator()(const ElementType *a, const size_t b_idx, size_t size,
                               ResultType /*worst_dist*/ = static_cast<ResultType>(-1)) const
  {
    ResultType dist = ResultType(0);
    for (size_t i = 0; i < size; i++) {
      ResultType diff = std::abs(a[i] - data_source.kdtree_get_pt(b_idx, i));
      if (diff > dist) dist = diff; // max{|a[i] - b[i]|}
    }
    return dist;
  }

  // --- accum_dist: case 1) both arguments are scalar arithmetic types (e.g., bbox low/high) ---
  template <class U, class V>
  inline typename std::enable_if<
    std::is_arithmetic<U>::value && std::is_arithmetic<V>::value,
    ResultType>::type
  accum_dist(const U &a, const V &b, size_t /*size*/) const
  {
    // For scalar vs scalar, Chebyshev distance is just abs difference
    return static_cast<ResultType>( std::abs(a - b) );
  }

  // -- accum_dist: case 2) at least one argument is array-like (pointer, vector, etc.) ---
  template <class U, class V>
  inline typename std::enable_if<
    !(std::is_arithmetic<U>::value && std::is_arithmetic<V>::value),
    ResultType>::type
  accum_dist(const U &a, const V &b, size_t size) const
  {
    ResultType dist = ResultType(0);
    for (size_t i = 0; i < size; i++) {
      // assume a[i] and b[i] are valid expressions (pointer-like or indexable)
      ResultType diff = std::abs(a[i] - b[i]);
      if (diff > dist) dist = diff;
    }
    return dist;
  }
};


// --- k-d tree with exact Chebyshev pruning ---
// nanoflann's searchLevel() combines the per-dimension cell offsets by SUMMING them
// (mindist + cut_dist - dst), which is a lower bound of the cell distance only for
// additive metrics (L1, squared L2). For L-infinity the cell distance is the MAXIMUM
// of the offsets; the sum over-estimates it, so cells that contain true neighbours are
// pruned and kNN radii come out too large (~9% of points in 2D, more in higher dims).
// This class hides findNeighbors() with a search that uses max() instead.
template <class T, class DataSource, typename IndexType = size_t>
class ChebyshevKDTree
  : public nanoflann::KDTreeSingleIndexAdaptor<Chebyshev_Adaptor<T, DataSource>, DataSource, -1, IndexType>
{
 public:
  using Base = nanoflann::KDTreeSingleIndexAdaptor<Chebyshev_Adaptor<T, DataSource>, DataSource, -1, IndexType>;
  using Base::Base;
  using NodePtr = typename Base::NodePtr;

  template <typename RESULTSET>
  bool findNeighbors(RESULTSET& result, const T* vec,
                     const nanoflann::SearchParameters& searchParams = {}) const
  {
    if (this->size(*this) == 0) return false;
    if (!this->root_node_)
      throw std::runtime_error("findNeighbors() called before building the index.");
    const size_t dim = this->dim_;
    std::vector<T> dists(dim, T(0));
    T mindist = T(0);
    for (size_t i = 0; i < dim; ++i) {
      if (vec[i] < this->root_bbox_[i].low)  dists[i] = this->root_bbox_[i].low - vec[i];
      if (vec[i] > this->root_bbox_[i].high) dists[i] = vec[i] - this->root_bbox_[i].high;
      if (dists[i] > mindist) mindist = dists[i];
    }
    searchLevelLinf(result, vec, this->root_node_, mindist, dists);
    if (searchParams.sorted) result.sort();
    return result.full();
  }

  // Same signature as nanoflann's knnSearch, routed through the exact search above.
  size_t knnSearch(const T* query_point, const size_t num_closest,
                   IndexType* out_indices, T* out_distances) const
  {
    nanoflann::KNNResultSet<T, IndexType> resultSet(num_closest);
    resultSet.init(out_indices, out_distances);
    findNeighbors(resultSet, query_point);
    return resultSet.size();
  }

  size_t radiusSearch(const T* query_point, const T& radius,
                      std::vector<nanoflann::ResultItem<IndexType, T>>& IndicesDists,
                      const nanoflann::SearchParameters& searchParams = {}) const
  {
    nanoflann::RadiusResultSet<T, IndexType> resultSet(radius, IndicesDists);
    findNeighbors(resultSet, query_point, searchParams);
    if (searchParams.sorted)
      std::sort(IndicesDists.begin(), IndicesDists.end(), nanoflann::IndexDist_Sorter());
    return resultSet.size();
  }

 private:
  template <class RESULTSET>
  bool searchLevelLinf(RESULTSET& result_set, const T* vec, const NodePtr node,
                       T mindist, std::vector<T>& dists) const
  {
    if ((node->child1 == nullptr) && (node->child2 == nullptr)) {
      T worst_dist = result_set.worstDist();
      for (auto i = node->node_type.lr.left; i < node->node_type.lr.right; ++i) {
        const IndexType accessor = this->vAcc_[i];
        T dist = this->distance_.evalMetric(vec, accessor, this->dim_);
        if (dist < worst_dist) {
          if (!result_set.addPoint(dist, accessor)) return false;
        }
      }
      return true;
    }

    const auto idx  = node->node_type.sub.divfeat;
    const T    val  = vec[idx];
    const T    diff1 = val - node->node_type.sub.divlow;
    const T    diff2 = val - node->node_type.sub.divhigh;

    NodePtr bestChild, otherChild;
    T cut_dist;
    if ((diff1 + diff2) < 0) {
      bestChild  = node->child1;
      otherChild = node->child2;
      cut_dist   = std::abs(val - node->node_type.sub.divhigh);
    } else {
      bestChild  = node->child2;
      otherChild = node->child1;
      cut_dist   = std::abs(val - node->node_type.sub.divlow);
    }

    if (!searchLevelLinf(result_set, vec, bestChild, mindist, dists)) return false;

    const T dst = dists[idx];
    dists[idx] = cut_dist;
    T otherdist = T(0);
    for (size_t i = 0; i < dists.size(); ++i)
      if (dists[i] > otherdist) otherdist = dists[i];
    if (otherdist <= result_set.worstDist()) {
      if (!searchLevelLinf(result_set, vec, otherChild, otherdist, dists)) return false;
    }
    dists[idx] = dst;
    return true;
  }
};

