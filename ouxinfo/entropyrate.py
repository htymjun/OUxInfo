import numpy as np
from ._core import shannon_entropy


def entropy_rate(x, tau=1, dt=1.e0, k=5):
  """Compute entropy rate.

  Parameters
  ----------
  x : ndarray of shape (N), (N, dim) or (N, dim, var)
      Input data. Must be float64.
  tau : int, optional
      Time delay (in samples). Default 1.
  dt : float, optional
      Physical time step. Default 1.0.
  k : int, optional
      Number of nearest neighbors. Default 5.

  Returns
  -------
  float
      Entropy rate.
  """
  if x.ndim == 1:
    x = x.reshape(-1,1)
  if x.ndim == 2:
    H1 = shannon_entropy(x[:-tau], k=k)
    H2 = shannon_entropy(x[tau:],  k=k)
    return (H2 - H1) / dt
  elif x.ndim == 3:
    n_vars = x.shape[2]
    dHdt = np.zeros(n_vars)
    for i in range(n_vars):
      x_var = x[:,:,i]
      H1 = shannon_entropy(x_var[:-tau], k=k)
      H2 = shannon_entropy(x_var[tau:],  k=k)
      dHdt[i] = (H2 - H1) / dt
    return dHdt
  else:
    raise ValueError(f"Unsupported array dimension: {x.ndim}. Input must be 1D, 2D, or 3D.")
