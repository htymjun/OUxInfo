import numpy as np
import matplotlib.pyplot as plt
from ._core import mutual_info


def myParams(fontsize=20):
  plt.rcParams['mathtext.fontset'] = 'stix'
  plt.rcParams['xtick.direction']  = 'in'
  plt.rcParams['ytick.direction']  = 'in'
  plt.rcParams['font.size']        = fontsize
  try:
    plt.rcParams['font.family']    = 'Times New Roman'
  except:
    plt.rcParams['font.family']    = 'Liberation Serif'


def auto_mutual_info(x, taumax, k=5):
  """
  Calculates mutual information while shifting data,
  and returns the first lag (tau) where mutual information reaches a local minimum.
 
  Parameters:
  -----------
  x : array-like
    1D input data.
  taumax : int
    Maximum lag to search.
  k : int, default=5
    Number of nearest neighbors for the kNN method.
 
  Returns:
  --------
  tau_list : np.ndarray
    Array of lags from 0 to taumax.
  mi_list : np.ndarray
    Array of mutual information values corresponding to each tau.
  optimal_tau : int or None
    The first lag where mutual information reaches a local minimum. 
    Returns None if no local minimum is found.
  """
  x = np.ascontiguousarray(x)
  tau_list = np.arange(1, taumax + 1, 1)
  mi_list = []
  # Calculate mutual information for each tau
  for tau in tau_list:
    # Shift data by the time delay tau
    x_current = x[:-tau].reshape(-1, 1)
    x_delayed = x[tau:].reshape(-1, 1)
    # Compute I(X_t ; X_{t+tau})
    mi = mutual_info(x_current, x_delayed, k=k)
    mi_list.append(mi)
  mi_list = np.array(mi_list)
  # Search for the first local minimum
  optimal_tau = None
  for i in range(1, len(mi_list) - 1):
    if mi_list[i] < mi_list[i - 1] and mi_list[i] < mi_list[i + 1]:
      optimal_tau = i
      break
  return np.array(tau_list), np.array(mi_list), optimal_tau
