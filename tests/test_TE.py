import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from ouxinfo import transfer_entropy


def make_linear_driver(alpha, N, seed=0):
  rng = np.random.default_rng(seed)
  x = rng.standard_normal(N + 1)
  eps = rng.standard_normal(N + 1)
  y = np.empty(N + 1)
  y[0] = eps[0]
  for t in range(1, N + 1):
    y[t] = alpha * x[t - 1] + eps[t]
  return x[1:].reshape(-1, 1), y[1:].reshape(-1, 1)


def test_transfer_entropy_monotone_in_coupling():
  # Stronger coupling should produce a larger directional asymmetry TE(x→y) - TE(y→x)
  N = 10000
  alphas = [0.5e0, 1.0e0, 1.5e0, 2.0e0]
  asym_prev = -np.inf
  for alpha in alphas:
    x, y = make_linear_driver(alpha, N, seed=42)
    TE_xy = transfer_entropy(x, y, k=5, tau=1, m=1, lag=1, trial=0)
    TE_yx = transfer_entropy(y, x, k=5, tau=1, m=1, lag=1, trial=0)
    asym = TE_xy - TE_yx
    assert asym > asym_prev, f"alpha={alpha:.2f}: asymmetry {asym:.4f} not > prev {asym_prev:.4f}"
    asym_prev = asym


def test_transfer_entropy_causal_direction():
  N = 10000
  for i, alpha in enumerate(np.linspace(1.0e0, 1.5e0, 5)):
    x, y = make_linear_driver(alpha, N, seed=i)
    TE_xy = transfer_entropy(x, y, k=5, tau=1, m=1, lag=1, trial=0)
    TE_yx = transfer_entropy(y, x, k=5, tau=1, m=1, lag=1, trial=0)
    assert TE_xy > TE_yx + 0.05e0, f"alpha={alpha:.2f}: TE(x→y)={TE_xy:.4f}, TE(y→x)={TE_yx:.4f}"


def test_transfer_entropy_knn_k_dependence():
  alpha = 1.e0
  N = 10000
  x, y = make_linear_driver(alpha, N)
  for k in range(3, 16):
    TE_xy = transfer_entropy(x, y, k=k, tau=1, m=1, lag=1, trial=0)
    TE_yx = transfer_entropy(y, x, k=k, tau=1, m=1, lag=1, trial=0)
    assert TE_xy > TE_yx + 0.05e0, f"k={k}: TE(x→y)={TE_xy:.4f}, TE(y→x)={TE_yx:.4f}"


def _analytical_te_gaussian_ar(a, b, c_xy, sigma=1.0):
  A = np.array([[a, 0.0], [c_xy, b]])
  Q = np.array([[sigma**2, 0.0], [0.0, sigma**2]])
  Sigma = solve_discrete_lyapunov(A, Q)
  S_yy = Sigma[1, 1]
  S_xy = Sigma[0, 1]
  cov_y_next_y = b * S_yy + c_xy * S_xy
  var_y_given_yt = S_yy - cov_y_next_y**2 / S_yy
  return 0.5 * np.log(var_y_given_yt / sigma**2)


def _gaussian_ar_system(nt, a, b, c_xy, sigma, rng):
  x = np.zeros(nt + 1)
  y = np.zeros(nt + 1)
  x[0] = rng.normal(0.0, sigma)
  y[0] = rng.normal(0.0, sigma)
  for i in range(nt):
    x[i + 1] = a * x[i] + rng.normal(0.0, sigma)
    y[i + 1] = b * y[i] + c_xy * x[i] + rng.normal(0.0, sigma)
  return x, y


def test_transfer_entropy_vs_analytical():
  nt = 500
  n_trials = 50
  a, b, sigma = 0.5, 0.5, 1.0
  for c_xy in [0.3, 0.5, 0.7]:
    te_true = _analytical_te_gaussian_ar(a, b, c_xy, sigma)
    rng = np.random.default_rng(int(c_xy * 100))
    te_sum = 0.0
    for _ in range(n_trials):
      x, y = _gaussian_ar_system(nt, a, b, c_xy, sigma, rng)
      te_sum += transfer_entropy(x.reshape(-1, 1), y.reshape(-1, 1),
                                 k=5, tau=1, m=1, lag=1, trial=1)
    te_est = te_sum / n_trials
    tol = 0.1e0 * te_true
    assert np.isclose(te_est, te_true, atol=tol), (
      f"c_xy={c_xy}: est={te_est:.4f}, true={te_true:.4f}, tol={tol:.4f}")
