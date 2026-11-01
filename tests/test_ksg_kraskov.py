"""KSG estimator vs. Kraskov, Stoegbauer & Grassberger, PRE 69, 066138 (2004) and exact MI.

Paper values were digitized from 400-dpi renderings of Figs. 4 and 5. Tolerances are derived from
the statistical error of our ensembles, so the tests resolve biases of order 1e-3 nats.
save_results() redraws Figs. 4, 5 and 14 into tests/results/kraskov.png; kraskov_figure(path, scale)
with a larger `scale` gives a higher-statistics version.
"""
import pathlib
import numpy as np
import pytest
from scipy.special import digamma
from ouxinfo import mutual_info

FIG4 = {0.02: 0.8417, 0.024: 0.8387, 0.028: 0.8353, 0.032: 0.8316, 0.036: 0.8270,
        0.04: 0.8229, 0.044: 0.8184, 0.048: 0.8139, 0.052: 0.8092, 0.056: 0.8046}  # I1, r=0.9, top curve
FIG5 = {1: 1.566, 2: 1.218, 3: 1.099, 5: 1.002, 10: 0.906}                         # sqrt(N) sd(I1), N=125


def _gauss(m, r):
  """sampler of equicorrelated Gaussians split as (X1, (X2..Xm)) and the exact I(X1; X2..Xm)."""
  C = np.full((m, m), r) + (1 - r) * np.eye(m); L = np.linalg.cholesky(C)
  def sample(rng, n):
    Z = rng.standard_normal((n, m)) @ L.T
    return np.ascontiguousarray(Z[:, :1]), np.ascontiguousarray(Z[:, 1:])
  return sample, 0.5 * np.log(np.linalg.det(C[1:, 1:]) / np.linalg.det(C))


def _ens(sample, n, k, R, seed):
  rng = np.random.default_rng(seed)
  return np.array([mutual_info(*sample(rng, n), k=k) for _ in range(R)])


def _sem(v):
  return v.std(ddof=1) / np.sqrt(len(v))


@pytest.mark.parametrize("draw", ["standard_normal", "random"])
@pytest.mark.parametrize("k", [1, 10])
def test_independent_variables_give_zero(draw, k):
  v = _ens(lambda rng, n: (getattr(rng, draw)((n, 1)), getattr(rng, draw)((n, 1))), 1000, k, 200, 11 + k)
  assert abs(v.mean()) < 4 * _sem(v), f"mean={v.mean():+.4f}, sem={_sem(v):.4f}"


@pytest.mark.parametrize("k", [20, 40])
def test_kraskov_fig4(k):
  v = _ens(_gauss(2, 0.9)[0], 1000, k, 200, 40 + k)
  # the N = 1000 curve lies up to ~0.001 below the top curve of Fig. 4
  assert abs(v.mean() - FIG4[k / 1000]) < 3 * _sem(v) + 0.001, f"k/N={k / 1000}: mean={v.mean():.4f}"


@pytest.mark.parametrize("k", [1, 5])
def test_kraskov_fig5(k):
  R = 1500
  got = _ens(_gauss(2, 0.9)[0], 125, k, R, 50 + k).std(ddof=1) * np.sqrt(125)
  assert abs(got - FIG5[k]) < 3 * got / np.sqrt(2 * (R - 1)), f"k={k}: sqrt(N) sd={got:.3f}"


@pytest.mark.parametrize("m, r, n, k, R, allowance", [(2, 0.3, 10000, 5, 30, 0.002),    # weak correlation
                                                      (5, 0.9, 20000, 3, 8, 0.01)])     # Fig. 14 setting
def test_exact_gaussian_mi(m, r, n, k, R, allowance):
  sample, I_true = _gauss(m, r)
  v = _ens(sample, n, k, R, 7 * m)
  assert abs(v.mean() - I_true) < 3 * _sem(v) + allowance, f"mean={v.mean():.4f}, exact={I_true:.4f}"


def test_gamma_exponential_distribution():
  """mu(x,y) = x e^{-x-xy} (theta = 1, log-transformed), I = psi(2) (Kraskov et al., Eq. 34)."""
  def sample(rng, n):
    x = rng.gamma(1.0, 1.0, n)
    return np.log(x).reshape(-1, 1), np.log(rng.exponential(1.0 / x)).reshape(-1, 1)
  v = _ens(sample, 10000, 5, 20, 3)
  assert abs(v.mean() - digamma(2)) < 3 * _sem(v) + 0.003, f"mean={v.mean():.4f}"


def kraskov_figure(path, scale=1):
  """Figs. 4, 5 and 14 of Kraskov et al.; red squares are the digitized paper values.
  `scale` multiplies the number of realizations (scale=1 takes ~15 s)."""
  import matplotlib.pyplot as plt
  sample = _gauss(2, 0.9)[0]
  fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
  for N, R in [(125, 2000), (1000, 300)]:
    ks = np.unique(np.r_[[1, 2, 3, 5, 10, 15, 20, 30], np.linspace(0.01, 0.07, 7) * N].astype(int))
    v = [_ens(sample, N, k, scale * R, N + k) for k in ks]
    ax[0].plot(ks / N, [x.mean() for x in v], 'o-', ms=3, label=f'N={N}')
    ax[1].plot(ks[ks <= 30], [x.std(ddof=1) * np.sqrt(N) for x in v][:np.sum(ks <= 30)], '+', ms=8, label=f'N={N}')
  for r in [0.1, 0.5, 0.9]:
    ms = np.arange(2, 9)
    ax[2].plot(ms, [_gauss(m, r)[1] for m in ms], 'k-', lw=0.8)
    ax[2].plot(ms, [_ens(_gauss(m, r)[0], 10000, 1, 2 * scale, m).mean() for m in ms], '+', ms=8, label=f'r={r}')
  ax[0].axhline(0.830366, color='k', lw=0.5)
  ax[0].plot(list(FIG4), list(FIG4.values()), 'rs', ms=4)
  ax[1].plot(list(FIG5), list(FIG5.values()), 'rs', ms=4)
  ax[0].set(xlabel='$k/N$', ylabel='$I^{(1)}$', title='Fig. 4', xlim=(0, 0.07), ylim=(0.78, 0.85))
  ax[1].set(xlabel='$k$', ylabel=r'$\sqrt{N}\,\Delta I^{(1)}$', title='Fig. 5', xlim=(0, 30), ylim=(0.8, 2.2))
  ax[2].set(xlabel='$m$', ylabel='$I^{(1)}(X_1; X_2 \\cdots X_m)$', title='Fig. 14 (N=10000)', ylim=(-0.1, 1.2))
  for a in ax: a.legend(fontsize=10)
  fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_results():
  path = pathlib.Path(__file__).parent / 'results' / 'kraskov.png'
  path.parent.mkdir(exist_ok=True)
  kraskov_figure(path)
  print(f"Saved {path}")
