import numpy as np
from scipy.special import psi, gamma
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def reshape_matrix(X):
  if X.ndim == 1:
    dx = 1.e0
    X  = X.reshape((-1,1))
  else:
    dx = X.shape[1]
  return X, dx


def shannon_entropy(X, k=3, Thei=1, Z=None):
  # Kozachenko Leonenko
  N = X.shape[0]
  X, dx = reshape_matrix(X)
  eps = np.zeros(N)
  tree_X = KDTree(X)
  # volume of unit ball in d*n
  Cdx = np.pi**(0.5e0*dx) / gamma(1.e0 + 0.5e0 * dx) / 2.e0**dx
  for i in range(N):
    distance, _ = tree_X.query(X[i], k=k+1, p=np.inf)
    eps[i] = 2.e0 * distance[-1]
  H = -psi(k) + psi(N) + np.log(Cdx) + dx * np.mean(np.log(eps))
  return H


def test_sigma():
  sigma = np.linspace(1.e0, 5.e0, 50)
  H_t = np.zeros(len(sigma))
  H_n = np.zeros(len(sigma))
  for i, s in enumerate(sigma):
    x = np.random.normal(0.e0, s, 500)
    H_t[i] = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))
    H_n[i] = shannon_entropy(x, k=5)
  plt.figure(figsize=(6,6))
  plt.plot(sigma, H_t, color='black')
  plt.plot(sigma, H_n, "o", color='blue')
  plt.xlabel(r'$\sigma$', fontsize=18, style='italic')
  plt.ylabel("H(X)", fontsize=18)
  plt.show()

  
def test_k():
  K = np.arange(1,50)
  s = 1.e0
  x = np.random.normal(0.e0, s, 500)
  H_t = np.zeros(len(K))
  H_n = np.zeros(len(K))
  for i, k in enumerate(K):
    H_t[i] = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))
    H_n[i] = shannon_entropy(x, k=k)
  plt.figure(figsize=(6,6))
  plt.plot(K, H_t, color='black')
  plt.plot(K, H_n, "o", color='blue')
  plt.xlabel("k", fontsize=18, style='italic')
  plt.ylabel("H(X)", fontsize=18)
  plt.show()


#test_sigma()
#test_k()

