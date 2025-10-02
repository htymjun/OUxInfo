import numpy as np
from shannon_entropy_cpp import shannon_entropy, KL_div
import time
from scipy.spatial import KDTree
from scipy.special import gamma

def kl_div(X, Y, k=3):
  """
  Pérez-Cruz 2008
  Parameters
  ----------
  X : ndarray of shape (n, d)
  Y : ndarray of shape (m, d)
  k : int
  Returns
  -------
  D_hat : float
  """
  n, d = X.shape
  m, _ = Y.shape
  tree_X = KDTree(X)
  tree_Y = KDTree(Y)
  r_x, _ = tree_X.query(X, k+1, p=np.inf)
  r = r_x[:,-1]
  s_y, _ = tree_Y.query(X, k, p=np.inf)
  s = s_y[:,-1]
  vol_unit_ball = (np.pi ** (d / 2)) / gamma(d / 2 + 1)
  D_hat = d * np.mean(np.log(s / r)) + np.log(m / (n - 1.0))
  return max(D_hat, 0.e0)

N = 10000
s = 1.e0
x = np.random.normal(0.e0, s, N)
Ht = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))

var_x = 1.e0
var_y = 9.e0
cov   = 1.e0
mean  = (0.e0, 0.e0)
rho   = cov / (np.sqrt(var_x * var_y)) # corr coef
Cov   = [[var_x, cov], \
         [cov, var_y]]
x = np.random.multivariate_normal(mean, Cov, N)

Dt = np.log(np.sqrt(var_y/var_x)) + var_x / (2.e0 * var_y) - 0.5e0
t1 = time.time()
Dn = kl_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
t2 = time.time()
Dc = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
t3 = time.time()
print("Theoretical:", Dt)
print("python:", Dn)
print("c++:", Dc)
print("Elapsed time py:", t2-t1)
print("Elapsed time c++:", t3-t2)
