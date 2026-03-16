import numpy as np
from scipy.special import psi, gamma
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from ouxinfo import mutual_info


def reshape_matrix(X):
  if X.ndim == 1:
    dx = 1.e0
    X  = X.reshape((-1,1))
  else:
    dx = X.shape[1]
  return X, dx


def mutual_info_py(X, Y, k=5, Thei=10):
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


def embedding_entropy(x, y, p=1, tau=1, k=5, Thei=None):
  '''
  x   : ndarray, shape (T) time series
  y   : ndarray, shape (T) time series
  p   : int, order of the model to estimate causality
  tau : int, time delay for embedding
  k   : int, k-th nearest neighbor number
  Thei: int, half-length of Theiler correction window
  out : float, embedding entropy y->x
  '''
  
  if Thei is None or Thei < p * tau:
    Thei = p * tau
  
  if x.ndim == 1:
    dx = 1
    x  = x.reshape(1,-1)
  else:
    dx = x.shape[0]
  if y.ndim == 1:
    dy = 1
    y  = y.reshape(1,-1)
  else:
    dy = y.shape[0]
 
  T = x.shape[1]

  X = np.hstack([x[:,(p-i)*tau:T-i*tau].T for i in range(p + 1)])
  Y = np.hstack([y[:,(p-i)*tau:T-i*tau].T for i in range(1, p + 1)])
  N = T - p * tau

  XNN = np.zeros((N, X.shape[1] * (dx * (p + 1) + 1)))
  
  for i in range(N):
    idx = np.ones(N, dtype=bool)
    idx[max(0, i - Thei):min(i + Thei + 1, N)] = False
    idx[i] = True
    
    temp_X = X[idx]
    tree_X = KDTree(temp_X)
    
    _, pnn_idx = tree_X.query(X[i], k=p + 2, workers=-1)
    XNN[i] = temp_X[pnn_idx].flatten()
  
  X_pass = np.ascontiguousarray(Y[:N],   dtype=np.float64)
  Y_pass = np.ascontiguousarray(XNN[:N], dtype=np.float64)
  EE = mutual_info_py(Y[:N], XNN[:N], k=k, Thei=Thei)
  #EE = mutual_info(Y_pass, X_pass, k=k, Thei=Thei)
  return EE


def embedding_entropy_surrogate(x, y, p=1, tau=1, k=5, Thei=10, trial=10):
  # informtion flow y -> x
  EE  = embedding_entropy(x, y,  p=p, tau=tau, k=k, Thei=Thei)
  EEs = 0.e0
  for _ in range(trial):
    ys = np.random.permutation(y)
    EEs += embedding_entropy(x, ys, p=p, tau=tau, k=k, Thei=Thei)
  return EE - EEs / trial

