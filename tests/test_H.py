import numpy as np
from ouxinfo import shannon_entropy


def theoretical_entropy_gaussian(sigma):
  return 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * sigma**2))


def test_entropy_gaussian_sigma_dependence():
  np.random.seed(0)
  N = 10000
  S = np.linspace(1.e0, 5.e0, 10)
  # for standard deviation
  for s in S:
    x = np.random.normal(0.e0, s, N)
    H_true = theoretical_entropy_gaussian(s)
    H_est = shannon_entropy(x.reshape(-1, 1), k=5)
    tol = 0.05e0 * H_true
    assert np.isclose(H_est, H_true, atol=tol), f"sigma={s}, est={H_est}, true={H_true}"


def test_entropy_knn_k_dependence():
  np.random.seed(0)
  N = 10000
  x = np.random.normal(0.e0, 1.e0, N)
  H_true = theoretical_entropy_gaussian(1.e0)
  for k in range(3, 20):
    H_est = shannon_entropy(x.reshape(-1, 1), k=k)
    tol = 0.05e0 * H_true
    assert np.isclose(H_est, H_true, atol=tol), f"k={k}, est={H_est}, true={H_true}"

