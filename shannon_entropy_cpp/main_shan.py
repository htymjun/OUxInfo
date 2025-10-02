import numpy as np
from shannon_entropy_cpp import shannon_entropy
from scipy.special import psi, gamma
from scipy.spatial import KDTree
import time

def count_points1(tree_X, tree_XY, X, XY, k, Thei=1):
  N   = X.shape[0]
  eps = np.zeros(N)
  nx  = np.zeros(N)
  for i in range(N):
    distance, _ = tree_XY.query(XY[i], k=k+1, p=np.inf)
    eps[i] = 2.e0 * distance[-1]
    idx = tree_X.query_ball_point(X[i], 0.5e0 * eps[i], p=np.inf)
    idx = [j for j in idx if abs(j - i) >= Thei]
    nx[i] = max(len(idx), 1.e0) # nx must be greater than 0 because psi(0) = -inf
  return nx, eps


def reshape_matrix(X):
  if X.ndim == 1:
    dx = 1.e0
    X  = X.reshape((-1,1))
  else:
    dx = X.shape[1]
  return X, dx


def shannon_entropy_py(X, k=3, Thei=1, Z=None):
  # Kozachenko Leonenko
  N = X.shape[0]
  X, dx = reshape_matrix(X)
  eps = np.zeros(N)
  tree_X = KDTree(X)
  # volume of unit ball in d*n
  Cdx = np.pi**(0.5e0*dx) / gamma(1.e0 + 0.5e0 * dx) / 2.e0**dx
  if Z is not None:
    Z, _ = reshape_matrix(Z)
    nx, eps = count_points1(tree_X, KDTree(Z), X, Z, k=k, Thei=Thei)
    H = np.mean(-psi(nx + 1.e0) + dx * np.log(eps)) + psi(N) + np.log(Cdx)
  else:
    for i in range(N):
      distance, _ = tree_X.query(X[i], k=k+1, p=np.inf)
      eps[i] = 2.e0 * distance[-1]
    H = -psi(k) + psi(N) + np.log(Cdx) + dx * np.mean(np.log(eps))
  return H

N = 1000
s = 1.e0
x = np.random.normal(0.e0, s, N)
Ht = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))

t1 = time.time()
Hn = shannon_entropy_py(x, k=3)
t2 = time.time()
Hc = shannon_entropy(x.reshape(-1,1), k=3)
t3 = time.time()

print("Theoretical:", Ht)
print("python:", Hn)
print("c++:", Hc)
print("Elapsed time python:", t2-t1)
print("Elapsed time c++:", t3-t2)
