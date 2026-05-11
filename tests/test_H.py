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

  # panel 1: H vs sigma
  S = np.linspace(1.e0, 5.e0, 10)
  H_true_s = [theoretical_entropy_gaussian(s) for s in S]
  H_est_s  = [shannon_entropy(np.random.normal(0.e0, s, N).reshape(-1, 1), k=5) for s in S]

  # panel 2: H vs k
  x_k = np.random.normal(0.e0, 1.e0, N)
  H_true_k = theoretical_entropy_gaussian(1.e0)
  ks = list(range(3, 20))
  H_est_k  = [shannon_entropy(x_k.reshape(-1, 1), k=k) for k in ks]

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  axes[0].plot(S, H_true_s, color='black', linestyle='solid', label='True')
  axes[0].plot(S, H_est_s,  color='blue',  linestyle='none', marker='o', label='Estimated')
  axes[0].set_xlabel(r'$\sigma$')
  axes[0].set_ylabel(r'$H$')
  axes[0].legend(frameon=False)

  axes[1].axhline(H_true_k, color='black', linestyle='solid', label='True')
  axes[1].plot(ks, H_est_k, color='blue', linestyle='none', marker='o', label='Estimated')
  axes[1].set_xlabel(r'$k$')
  axes[1].set_ylabel(r'$H$')
  axes[1].legend(frameon=False)

  plt.tight_layout()
  plt.savefig(_results / 'H.png', dpi=150, bbox_inches='tight')
  plt.close()
  print(f'Saved {_results / "H.png"}')


if __name__ == '__main__':
  save_results()

