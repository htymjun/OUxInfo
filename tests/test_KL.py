import numpy as np
from ouxinfo import KL_div


def theoretical_kl_gaussian(sigma_x, sigma_y):
  return np.log(sigma_y / sigma_x) + sigma_x**2 / (2.e0 * sigma_y**2) - 0.5e0


def test_kl_div_gaussian_variance_ratio_dependence():
  np.random.seed(0)
  N = 10000
  sigma_x = 1.e0
  for sigma_y in [2.e0, 3.e0, 4.e0, 5.e0]:
    x = np.random.normal(0.e0, sigma_x, N).reshape(-1, 1)
    y = np.random.normal(0.e0, sigma_y, N).reshape(-1, 1)
    KL_true = theoretical_kl_gaussian(sigma_x, sigma_y)
    KL_est = KL_div(x, y, k=5)
    tol = 0.05e0 * KL_true
    assert np.isclose(KL_est, KL_true, atol=tol), f"sigma_y={sigma_y}, est={KL_est:.4f}, true={KL_true:.4f}"


def test_kl_div_knn_k_dependence():
  np.random.seed(0)
  N = 10000
  sigma_x, sigma_y = 1.e0, 3.e0
  x = np.random.normal(0.e0, sigma_x, N).reshape(-1, 1)
  y = np.random.normal(0.e0, sigma_y, N).reshape(-1, 1)
  KL_true = theoretical_kl_gaussian(sigma_x, sigma_y)
  for k in range(3, 20):
    KL_est = KL_div(x, y, k=k)
    tol = 0.05e0 * KL_true
    assert np.isclose(KL_est, KL_true, atol=tol), f"k={k}, est={KL_est:.4f}, true={KL_true:.4f}"


def test_kl_div_identical_distribution():
  np.random.seed(0)
  N = 10000
  x = np.random.normal(0.e0, 1.e0, N).reshape(-1, 1)
  y = np.random.normal(0.e0, 1.e0, N).reshape(-1, 1)
  KL_est = KL_div(x, y, k=5)
  assert np.isclose(KL_est, 0.e0, atol=0.05e0), f"est={KL_est:.4f}, expected≈0"
