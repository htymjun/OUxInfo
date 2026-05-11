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


def save_results():
  import pathlib
  import matplotlib.pyplot as plt
  _results = pathlib.Path(__file__).parent / 'results'
  _results.mkdir(exist_ok=True)
  plt.rcParams['font.family'] = 'Times New Roman'
  plt.rcParams['mathtext.fontset'] = 'stix'
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  plt.rcParams['font.size'] = 20

  N = 10000
  np.random.seed(0)
  sigma_x = 1.e0

  # panel 1: KL vs sigma_y
  sigma_ys = [2.e0, 3.e0, 4.e0, 5.e0]
  KL_true_s = [theoretical_kl_gaussian(sigma_x, sy) for sy in sigma_ys]
  KL_est_s  = [KL_div(np.random.normal(0.e0, sigma_x, N).reshape(-1, 1),
                      np.random.normal(0.e0, sy, N).reshape(-1, 1), k=5)
               for sy in sigma_ys]

  # panel 2: KL vs k
  sigma_y = 3.e0
  x_k = np.random.normal(0.e0, sigma_x, N).reshape(-1, 1)
  y_k = np.random.normal(0.e0, sigma_y, N).reshape(-1, 1)
  KL_true_k = theoretical_kl_gaussian(sigma_x, sigma_y)
  ks = list(range(3, 20))
  KL_est_k = [KL_div(x_k, y_k, k=k) for k in ks]

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  axes[0].plot(sigma_ys, KL_true_s, color='black', linestyle='solid', label='True')
  axes[0].plot(sigma_ys, KL_est_s,  color='blue',  linestyle='none', marker='o', label='Estimated')
  axes[0].set_xlabel(r'$\sigma_y$')
  axes[0].set_ylabel(r'$D_{\rm KL}$')
  axes[0].legend(frameon=False)

  axes[1].axhline(KL_true_k, color='black', linestyle='solid', label='True')
  axes[1].plot(ks, KL_est_k, color='blue', linestyle='none', marker='o', label='Estimated')
  axes[1].set_xlabel(r'$k$')
  axes[1].set_ylabel(r'$D_{\rm KL}$')
  axes[1].legend(frameon=False)

  plt.tight_layout()
  plt.savefig(_results / 'KL.png', dpi=150, bbox_inches='tight')
  plt.close()
  print(f'Saved {_results / "KL.png"}')


if __name__ == '__main__':
  save_results()
