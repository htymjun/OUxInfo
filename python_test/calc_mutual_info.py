import numpy as np
from scipy.special import psi, gamma
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm


def reshape_matrix(X):
  if X.ndim == 1:
    dx = 1.e0
    X  = X.reshape((-1,1))
  else:
    dx = X.shape[1]
  return X, dx


def mutual_info(X, Y, k=5, Thei=10):
  X, _ = reshape_matrix(X)
  Y, _ = reshape_matrix(Y)
  N = X.shape[0]
  nX, nY, nN = np.zeros(N), np.zeros(N), np.zeros(N)
  for i in range(N):
    idx = np.ones(N, dtype=bool)
    idx[max(0, i - Thei):min(i + Thei + 1, N)] = False
    tree_XY = KDTree(np.hstack((X[idx], Y[idx])))
    dist, _ = tree_XY.query(np.hstack((X[i], Y[i])), k=k, p=np.inf, workers=-1)
    half_epsilon_XYkNN = dist[-1]
    nX[i] = np.sum(cdist(X[idx], X[i].reshape(1,-1), metric='chebyshev') < half_epsilon_XYkNN)
    nY[i] = np.sum(cdist(Y[idx], Y[i].reshape(1,-1), metric='chebyshev') < half_epsilon_XYkNN)
    nN[i] = np.sum(idx)
  valid_idx = (nX > 0) & (nY > 0)
  I = psi(k) - np.mean(psi(nX[valid_idx] + 1)) - np.mean(psi(nY[valid_idx] + 1)) + np.mean(psi(nN[valid_idx] + 1))
  return I 

  
def test_mutual_info():
  # variance
  var_x = 9.e0
  var_y = 25.e0
  covariance = np.linspace(0.e0, 10.e0, 50)
  mean = (0.e0, 0.e0)
  It   = np.zeros(len(covariance))
  In   = np.zeros(len(covariance))
  for i, cov in tqdm(enumerate(covariance)):
    rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
    Cov  = [[var_x, cov], \
            [cov, var_y]]
    x = np.random.multivariate_normal(mean, Cov, 10000)
    It[i] = -0.5e0 * np.log(1.e0 - rho**2)
    In[i] = mutual_info(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10, Thei=1)
  plt.figure(figsize=(6,6))
  plt.plot(covariance, It, color='black')
  plt.plot(covariance, In, "o", color='blue')
  plt.xlabel(r'$\sigma_{xy}$', fontsize=18, style='italic')
  plt.ylabel("I(X;Y)", fontsize=18)
  plt.show()


test_mutual_info()

