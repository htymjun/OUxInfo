#pragma once
#include "nanoflann.hpp"
#include "point_cloud.hpp"
#include <type_traits>
#include <cmath>


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


typedef KDTreeSingleIndexAdaptor<
  //L2_Simple_Adaptor<double, PointCloud>,
  Chebyshev_Adaptor<double, PointCloud>,
  PointCloud,
  -1
> kd_tree_t;

