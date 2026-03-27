import numpy as np
from ._core import transfer_entropy


def backward_transfer_entropy(x, y, tau=1, m=1, lag=1, dt=1.e0, k=5, trial=0):
  '''
  Parameters
  ----------
  x     : ndarray (N, dim)
  y     : ndarray (N, dim)
  tau   : int, optional
          Length of time delay
  m     : int, optional
          Embedding dimension for y
  lag   : int, optional
          Time lag for embedding
  dt    : double, optional
          Physical time
  k     : int, optional
          Number of nearest neighbors.
  trial : int, optional
          The number of trials for surrogate analysis.
  Returns
  -------
  double
    backward transfer entropy.
  '''
  x_ = x.reshape(-1,1) if x.ndim == 1 else x
  y_ = y.reshape(-1,1) if y.ndim == 1 else y
  xb = np.ascontiguousarray(x_[::-1,:])
  yb = np.ascontiguousarray(y_[::-1,:])
  BTE = transfer_entropy(xb, yb, tau=tau, m=m, lag=lag, dt=dt, k=k, trial=1)
  return BTE

