import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from ouxinfo import information_flow


def _analytical_if_gaussian_ar(a, b, c_xy, tau=1, dt=1.0, sigma=1.0):
  A = np.array([[a, 0.0], [c_xy, b]])
  Q = np.array([[sigma**2, 0.0], [0.0, sigma**2]])
  Sigma = solve_discrete_lyapunov(A, Q)
  S_xx, S_yy = Sigma[0, 0], Sigma[1, 1]
  Gamma_tau = np.linalg.matrix_power(A, tau) @ Sigma
  rho_0   = Sigma[1, 0]     / np.sqrt(S_xx * S_yy)
  rho_tau = Gamma_tau[1, 0] / np.sqrt(S_xx * S_yy)
  I_0   = -0.5e0 * np.log(1.0e0 - rho_0**2)
  I_tau = -0.5e0 * np.log(1.0e0 - rho_tau**2)
  return (I_tau - I_0) / dt


def _gaussian_ar_system(nt, a, b, c_xy, sigma, rng):
  x = np.zeros(nt + 1)
  y = np.zeros(nt + 1)
  x[0] = rng.normal(0.0, sigma)
  y[0] = rng.normal(0.0, sigma)
  for i in range(nt):
    x[i + 1] = a * x[i] + rng.normal(0.0, sigma)
    y[i + 1] = b * y[i] + c_xy * x[i] + rng.normal(0.0, sigma)
  return x, y


def test_information_flow_causal_direction():
  N = 5000
  a, b, sigma = 0.5e0, 0.5e0, 1.0e0
  for c_xy in [0.3e0, 0.5e0, 0.7e0]:
    rng = np.random.default_rng(int(c_xy * 100))
    x, y = _gaussian_ar_system(N, a, b, c_xy, sigma, rng)
    IF_xy = information_flow(x.reshape(-1, 1), y.reshape(-1, 1), tau=1, dt=1.0, k=5)
    IF_yx = information_flow(y.reshape(-1, 1), x.reshape(-1, 1), tau=1, dt=1.0, k=5)
    assert IF_xy > IF_yx, f"c_xy={c_xy}: IF(x→y)={IF_xy:.4f}, IF(y→x)={IF_yx:.4f}"


def test_information_flow_monotone_in_coupling():
  N = 5000
  a, b, sigma = 0.5e0, 0.5e0, 1.0e0
  asym_prev = -np.inf
  for i, c_xy in enumerate(np.linspace(0.2e0, 0.7e0, 5)):
    rng = np.random.default_rng(i)
    x, y = _gaussian_ar_system(N, a, b, c_xy, sigma, rng)
    IF_xy = information_flow(x.reshape(-1, 1), y.reshape(-1, 1), tau=1, dt=1.0, k=5)
    IF_yx = information_flow(y.reshape(-1, 1), x.reshape(-1, 1), tau=1, dt=1.0, k=5)
    asym = IF_xy - IF_yx
    assert asym > asym_prev, f"c_xy={c_xy:.2f}: asym={asym:.4f} not > prev {asym_prev:.4f}"
    asym_prev = asym


def test_information_flow_vs_analytical():
  nt = 500
  n_trials = 50
  a, b, sigma = 0.5e0, 0.5e0, 1.0e0
  for c_xy in [0.3e0, 0.5e0, 0.7e0]:
    IF_true = _analytical_if_gaussian_ar(a, b, c_xy, tau=1, dt=1.0, sigma=sigma)
    rng = np.random.default_rng(int(c_xy * 100))
    IF_sum = 0.0e0
    for _ in range(n_trials):
      x, y = _gaussian_ar_system(nt, a, b, c_xy, sigma, rng)
      IF_sum += information_flow(x.reshape(-1, 1), y.reshape(-1, 1), tau=1, dt=1.0, k=5)
    IF_est = IF_sum / n_trials
    tol = 0.1e0 * IF_true
    assert np.isclose(IF_est, IF_true, atol=tol), (
      f"c_xy={c_xy}: est={IF_est:.4f}, true={IF_true:.4f}, tol={tol:.4f}")


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

  N = 5000
  a, b, sigma = 0.5e0, 0.5e0, 1.0e0
  c_xys = np.linspace(0.1e0, 0.8e0, 8)
  IF_true_vals, IF_xy_vals, IF_yx_vals = [], [], []
  for c_xy in c_xys:
    rng = np.random.default_rng(42)
    x, y = _gaussian_ar_system(N, a, b, c_xy, sigma, rng)
    IF_true_vals.append(_analytical_if_gaussian_ar(a, b, c_xy, tau=1, dt=1.0, sigma=sigma))
    IF_xy_vals.append(information_flow(x.reshape(-1, 1), y.reshape(-1, 1), tau=1, dt=1.0, k=5))
    IF_yx_vals.append(information_flow(y.reshape(-1, 1), x.reshape(-1, 1), tau=1, dt=1.0, k=5))

  plt.figure(figsize=(7, 6))
  plt.plot(c_xys, IF_true_vals, color='black', linestyle='solid',  label=r'$T_{x \to y}$ (true)')
  plt.plot(c_xys, IF_xy_vals,   color='blue',  linestyle='none', marker='o', label=r'$T_{x \to y}$')
  plt.plot(c_xys, IF_yx_vals,   color='red',   linestyle='none', marker='^', label=r'$T_{y \to x}$')
  plt.axhline(0.e0, color='gray', linestyle='dotted')
  plt.xlabel(r'$c_{xy}$')
  plt.ylabel('Information flow')
  plt.legend(frameon=False)
  plt.savefig(_results / 'IF.png', dpi=150, bbox_inches='tight')
  plt.close()
  print(f'Saved {_results / "IF.png"}')


if __name__ == '__main__':
  save_results()
