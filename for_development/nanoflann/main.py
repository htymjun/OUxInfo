import numpy as np
from scipy.spatial import KDTree
from nanoflann_knn import knn_search
import time

"""
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
"""

def KDTree_py_2D(x, y, k=5):
    
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y

    # 結合空間 (X,Y)
    xy = np.hstack((x, y))
    tree_xy = KDTree(xy)
    d_xy, i = tree_xy.query(xy, k=k+1, p=np.inf, workers=-1)
    dists   = d_xy[:, -1]
    indices = i[:, -1]
    return indices, dists 


N = 5
var_x = 9.e0
var_y = 25.e0
cov = 10
It = 0.0
In = 0.0
In2 = 0.0
Ic = 0.0
mean = (0.e0, 0.e0)
rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
Cov  = [[var_x, cov], \
            [cov, var_y]]
x = np.random.multivariate_normal(mean, Cov, N)

t1 = time.time()
indices_py, dists_py = KDTree_py_2D(x[:,0].reshape(-1,1),x[:,1].reshape(-1,1), k=3)
t2 = time.time()
X = np.stack([x[:,0], x[:,1]], axis=1)
indices_cp, dists_cp = knn_search(X, k=3)
t3 = time.time()
print("Indices_py:\n", indices_py,"\n", indices_cp)
print("Indices_c+:\n", dists_py,"\n", dists_cp)
print("Elapsed time:", t2-t1, t3-t2)

