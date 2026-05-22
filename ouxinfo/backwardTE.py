import numpy as np
from ._core import transfer_entropy


def backward_transfer_entropy(x, y, tau=1, m=1, lag=1, dt=1.e0, k=5, trial=0):
  """Compute backward transfer entropy from x to y.

  Parameters
  ----------
  x : ndarray of shape (N, dim)
      Source time series. Must be float64.
  y : ndarray of shape (N, dim)
      Target time series. Must be float64.
  tau : int, optional
      Time delay (in samples). Default 1.
  m : int, optional
      Embedding dimension for y. Default 1.
  lag : int, optional
      Time lag for embedding. Default 1.
  dt : float, optional
      Physical time step. Default 1.0.
  k : int, optional
      Number of nearest neighbors. Default 5.
  trial : int, optional
      Number of surrogate trials for significance testing. Default 0.

  Returns
  -------
  float
      Backward transfer entropy computed on time-reversed series.
  """
  x_ = x.reshape(-1,1) if x.ndim == 1 else x
  y_ = y.reshape(-1,1) if y.ndim == 1 else y
  xb = np.ascontiguousarray(x_[::-1,:])
  yb = np.ascontiguousarray(y_[::-1,:])
  BTE = transfer_entropy(xb, yb, tau=tau, m=m, lag=lag, dt=dt, k=k, trial=1)
  return BTE

