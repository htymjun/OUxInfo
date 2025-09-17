import numpy as np
from scipy.spatial import KDTree
import nanoflann_knn
import time


def KDTree_py(X, k=5):
  tree = KDTree(X)
  N = X.shape[0]
  dists   = np.zeros(N, dtype=np.float64)
  indices = np.zeros_like(dists)
  for itr, x in enumerate(X):
    d, i = tree.query(x, k=k, p=np.inf)
    dists[itr] = d[-1]
    indices[itr] = i[-1]
  return indices, dists


N = 10
s = 1.e0
x = np.random.normal(0.e0, s, N).astype(np.float64)
x = x.reshape(-1,1)


t1 = time.time()
indices_py, dists_py = KDTree_py(x, k=3)
t2 = time.time()
indices_cp, dists_cp = nanoflann_knn.knn_search(x, k=3)
t3 = time.time()
print("Indices:\n", dists_cp, dists_py)
print("Elapsed time:", t2-t1, t3-t2)

