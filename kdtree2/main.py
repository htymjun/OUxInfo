import numpy as np
from scipy.spatial import KDTree
from kdtree2 import kdtree_query
import time


def KDTree_py(x, k=5):
  tree = KDTree(x)
  d, _ = tree.query(x[0], k=k+1, p=np.inf)
  return d[-1]


N = 500000
s = 1.e0
x = np.random.normal(0.e0, s, N).astype(np.float32)
x = np.asfortranarray(x)
x = x.reshape(-1,1)


t1 = time.time()
dp = KDTree_py(x, k=5)
t2 = time.time()
df = kdtree_query(x, k=5)
t3 = time.time()
print("scipy:", dp, " kdtree2:", df)
print("Elapsed time:", t2-t1, t3-t2)

