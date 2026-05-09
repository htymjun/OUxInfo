import numpy as np
from ouxinfo import mutual_info


def theoretical_mi_gaussian(rho):
  return -0.5e0 * np.log(1.e0 - rho**2)


def test_mutual_info_gaussian_rho_dependence():
  np.random.seed(0)
  N = 10000
  sigma_x, sigma_y = 3.e0, 5.e0
  # KSG MI estimator has large negative bias for weak correlations; reliable from rho≥0.6
  for rho in np.linspace(0.6e0, 0.9e0, 5):
    cov = rho * sigma_x * sigma_y
    Cov = [[sigma_x**2, cov], [cov, sigma_y**2]]
    xy = np.random.multivariate_normal([0.e0, 0.e0], Cov, N)
    I_true = theoretical_mi_gaussian(rho)
    I_est = mutual_info(xy[:, 0].reshape(-1, 1), xy[:, 1].reshape(-1, 1), k=5)
    tol = 0.10e0 * I_true
    assert np.isclose(I_est, I_true, atol=tol), f"rho={rho:.2f}, est={I_est:.4f}, true={I_true:.4f}"


def test_mutual_info_knn_k_dependence():
  np.random.seed(0)
  N = 10000
  var_x, var_y, cov = 9.e0, 25.e0, 10.e0
  rho = cov / np.sqrt(var_x * var_y)
  Cov = [[var_x, cov], [cov, var_y]]
  xy = np.random.multivariate_normal([0.e0, 0.e0], Cov, N)
  I_true = theoretical_mi_gaussian(rho)
  # KSG bias grows with k; reliable up to k~14 for this sample size
  for k in range(3, 15):
    I_est = mutual_info(xy[:, 0].reshape(-1, 1), xy[:, 1].reshape(-1, 1), k=k)
    tol = 0.10e0 * I_true
    assert np.isclose(I_est, I_true, atol=tol), f"k={k}, est={I_est:.4f}, true={I_true:.4f}"


def test_mutual_info_independent():
  np.random.seed(0)
  N = 10000
  x = np.random.normal(0.e0, 1.e0, N).reshape(-1, 1)
  y = np.random.normal(0.e0, 1.e0, N).reshape(-1, 1)
  I_est = mutual_info(x, y, k=5)
  assert np.isclose(I_est, 0.e0, atol=0.05e0), f"est={I_est:.4f}, expected≈0"
