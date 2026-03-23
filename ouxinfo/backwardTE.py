import numpy as np
from ._core import transfer_entropy


def backward_transfer_entropy(x, y, tau=1, m=1, lag=1, dt=1.e0, k=5, trial=0):
  x_ = x.reshape(-1,1) if x.ndim == 1 else x
  y_ = y.reshape(-1,1) if y.ndim == 1 else y
  xb = np.ascontiguousarray(x_[::-1,:])
  yb = np.ascontiguousarray(y_[::-1,:])
  BTE = transfer_entropy(xb, yb, tau=tau, m=m, lag=lag, dt=dt, k=k, trial=1)
  return BTE

