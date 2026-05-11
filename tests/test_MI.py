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
  sigma_x, sigma_y = 3.e0, 5.e0

  # panel 1: MI vs rho
  rhos = np.linspace(0.6e0, 0.9e0, 5)
  I_true_r, I_est_r = [], []
  for rho in rhos:
    cov = rho * sigma_x * sigma_y
    Cov = [[sigma_x**2, cov], [cov, sigma_y**2]]
    xy = np.random.multivariate_normal([0.e0, 0.e0], Cov, N)
    I_true_r.append(theoretical_mi_gaussian(rho))
    I_est_r.append(mutual_info(xy[:, 0].reshape(-1, 1), xy[:, 1].reshape(-1, 1), k=5))

  # panel 2: MI vs k
  rho_k = 0.75e0
  cov_k = rho_k * sigma_x * sigma_y
  Cov_k = [[sigma_x**2, cov_k], [cov_k, sigma_y**2]]
  xy_k = np.random.multivariate_normal([0.e0, 0.e0], Cov_k, N)
  I_true_k = theoretical_mi_gaussian(rho_k)
  ks = list(range(3, 15))
  I_est_k = [mutual_info(xy_k[:, 0].reshape(-1, 1), xy_k[:, 1].reshape(-1, 1), k=k) for k in ks]

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  axes[0].plot(rhos, I_true_r, color='black', linestyle='solid', label='True')
  axes[0].plot(rhos, I_est_r,  color='blue',  linestyle='none', marker='o', label='Estimated')
  axes[0].set_xlabel(r'$\rho$')
  axes[0].set_ylabel(r'$I$')
  axes[0].legend(frameon=False)

  axes[1].axhline(I_true_k, color='black', linestyle='solid', label='True')
  axes[1].plot(ks, I_est_k, color='blue', linestyle='none', marker='o', label='Estimated')
  axes[1].set_xlabel(r'$k$')
  axes[1].set_ylabel(r'$I$')
  axes[1].legend(frameon=False)

  plt.tight_layout()
  plt.savefig(_results / 'MI.png', dpi=150, bbox_inches='tight')
  plt.close()
  print(f'Saved {_results / "MI.png"}')


if __name__ == '__main__':
  save_results()
