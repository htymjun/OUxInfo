"""Conditional information flow (optional Z argument of information_flow and
information_flow_causal_map).

  T_{X->Y|Z} = [I(X_t; Y_{t+tau} | Z_t) - I(X_t; Y_t | Z_t)] / (tau dt)

Checks, from exact to statistical:
  1. agreement with an exhaustive-distance KSG reference (all code paths),
  2. exact identities: Z = Y gives the transfer entropy, Z = (Y, W) the conditional
     transfer entropy, and a constant Z the unconditional information flow,
  3. agreement with closed-form values for linear Gaussian VAR(1) systems.
"""
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from ouxinfo import (information_flow, information_flow_causal_map, transfer_entropy,
                     conditional_mutual_info)
from test_knn_exact import _cmi


def _c(a):
  return np.ascontiguousarray(a)


def _cond_if_ref(x, y, z, tau, dt, k):
  m = len(x) - tau
  return (_cmi(x[:m], y[tau:], z[:m], k) - _cmi(x[:m], y[:m], z[:m], k)) / (tau * dt)


# ---------------------------------------------------------------------------
# Gaussian VAR(1): s_{t+1} = A s_t + e_t, e_t ~ N(0, I); s = (X, Y, W)
# ---------------------------------------------------------------------------

CHAIN  = np.array([[0.5, 0.0, 0.0], [0.0, 0.3, 0.9], [0.9, 0.0, 0.3]])  # X -> W -> Y
DRIVER = np.array([[0.5, 0.0, 0.6], [0.0, 0.5, 0.6], [0.0, 0.0, 0.7]])  # W -> X, W -> Y
DIRECT = np.array([[0.5, 0.0, 0.5], [0.4, 0.5, 0.5], [0.0, 0.0, 0.6]])  # X -> Y, W -> X, Y


def _simulate(A, n, rng, burn=500):
  s = np.zeros((n + burn, A.shape[0]))
  e = rng.standard_normal(s.shape)
  for t in range(n + burn - 1):
    s[t + 1] = A @ s[t] + e[t + 1]
  return s[burn:]


def _gaussian_cmi(V, a, b, c):
  ld = lambda ix: np.linalg.slogdet(V[np.ix_(ix, ix)])[1] if ix else 0.0
  return 0.5 * (ld(a + c) + ld(b + c) - ld(c) - ld(a + b + c))


def _exact_cond_if(A, tau, cond, dt=1.0):
  """T_{X->Y|Z} with Z = s_t[cond]; cond = [1] is the transfer entropy."""
  d = A.shape[0]
  S = solve_discrete_lyapunov(A, np.eye(d))
  C = np.linalg.matrix_power(A, tau) @ S                  # Cov(s_{t+tau}, s_t)
  V = np.block([[S, C.T], [C, S]])                        # (s_t, s_{t+tau})
  # I(X_t; Y_t | Z_t) = 0 when Y_t is part of Z_t
  second = 0.0 if 1 in cond else _gaussian_cmi(V, [0], [1], cond)
  return (_gaussian_cmi(V, [0], [d + 1], cond) - second) / (tau * dt)


# ---------------------------------------------------------------------------
# 1. brute-force agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("tau", [1, 3])
@pytest.mark.parametrize("dims", [(1, 1, 1), (1, 1, 2), (2, 1, 1), (1, 3, 2)])
def test_conditional_information_flow_matches_brute_force(dims, tau, k):
  dx, dy, dz = dims
  rng = np.random.default_rng(100 * k + 10 * tau + sum(dims))
  d = dx + dy + dz
  S = rng.standard_normal((600, d)) @ (np.eye(d) + 0.4 * rng.standard_normal((d, d)))
  x, y, z = _c(S[:, :dx]), _c(S[:, dx:dx + dy]), _c(S[:, dx + dy:])
  ref = _cond_if_ref(x, y, z, tau, 0.37, k)
  assert information_flow(x, y, tau=tau, dt=0.37, k=k, Z=z) == pytest.approx(ref, abs=1e-10)


@pytest.mark.parametrize("dx, dz, z_1d", [(1, 1, True), (1, 2, False), (2, 1, False), (2, 2, False)])
def test_conditional_causal_maps_match_brute_force(dx, dz, z_1d):
  rng = np.random.default_rng(10 * dx + dz)
  Nx, n, tau, dt, k = 4, 400, 2, 0.1, 3
  X = np.zeros((Nx, n, dx)); W = np.zeros((n, dz))
  for t in range(n - 1):
    W[t + 1]    = 0.6 * W[t] + rng.standard_normal(dz)
    X[0, t + 1] = 0.5 * X[0, t] + 0.7 * W[t, 0]  + rng.standard_normal(dx)
    X[1, t + 1] = 0.5 * X[1, t] + 0.6 * X[0, t]  + rng.standard_normal(dx)
    X[2, t + 1] = 0.5 * X[2, t] + 0.5 * W[t, -1] + rng.standard_normal(dx)
    X[3, t + 1] = 0.4 * X[3, t] + 0.6 * X[2, t]  + rng.standard_normal(dx)
  m = n - tau
  IF_ref = np.full((Nx, Nx), np.nan); dI_ref = IF_ref.copy()
  for i in range(Nx):
    for j in range(Nx):
      if i != j:
        IF_ref[j, i] = _cond_if_ref(X[i], X[j], W, tau, dt, k)
        dI_ref[i, j] = (_cmi(X[i, tau:], X[j, tau:], W[:m], k) - _cmi(X[i, :m], X[j, :m], W[:m], k)) / (tau * dt)
  Leak_ref = dI_ref - IF_ref - IF_ref.T

  X_in = _c(X[:, :, 0]) if dx == 1 else X
  Z_in = W[:, 0] if z_1d else W
  off = ~np.eye(Nx, dtype=bool)
  sparse = np.zeros((Nx, Nx), dtype=bool); sparse[0, 1] = sparse[3, 2] = True
  for full in (False, True):
    for n_threads in (1, 3):
      for mask in (None, off, sparse):
        out = information_flow_causal_map(X_in, tau, mask=mask, dt=dt, k=k,
                                          n_threads=n_threads, full=full, Z=Z_in)
        maps = out if full else (out,)
        sel = off if mask is None else (mask | mask.T)   # a pair is computed in both directions
        for got, ref in zip(maps, (IF_ref, Leak_ref, dI_ref)):
          np.testing.assert_allclose(got[sel], ref[sel], atol=1e-10, rtol=0)
          assert np.all(np.isnan(np.diag(got)))
          assert np.all(got[off & ~sel] == 0.0)


# ---------------------------------------------------------------------------
# 2. exact identities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("tau", [1, 3])
def test_conditioning_on_target_gives_transfer_entropy(tau, k):
  """I(X_t; Y_t | Y_t) = 0, so T_{X->Y|Y} is TE(X->Y); this compares the new CMI kernel
  against the independent one behind transfer_entropy."""
  s = _simulate(DIRECT, 3000, np.random.default_rng(tau + k))
  x, y, w = (_c(s[:, i:i + 1]) for i in range(3))
  dt = 0.25
  te = transfer_entropy(x, y, tau=tau, m=1, lag=1, dt=dt, k=k, trial=0)
  assert information_flow(x, y, tau=tau, dt=dt, k=k, Z=y) == pytest.approx(te, abs=1e-12)

  # embedding m = 2, lag = 2: Z_t = (Y_t, Y_{t-2})
  lag = 2
  te2 = transfer_entropy(x, y, tau=tau, m=2, lag=lag, dt=dt, k=k, trial=0)
  z2 = _c(np.hstack([y[lag:], y[:-lag]]))
  assert information_flow(_c(x[lag:]), _c(y[lag:]), tau=tau, dt=dt, k=k, Z=z2) == pytest.approx(te2, abs=1e-12)

  # conditional transfer entropy I(X_t; Y_{t+tau} | Y_t, W_t)
  m = len(x) - tau
  yw = _c(np.hstack([y, w]))
  cte = conditional_mutual_info(x[:m], _c(y[tau:]), _c(yw[:m]), k=k) / (tau * dt)
  assert information_flow(x, y, tau=tau, dt=dt, k=k, Z=yw) == pytest.approx(cte, abs=1e-12)

  IF_map = information_flow_causal_map(_c(s[:, :2].T), tau, dt=dt, k=k, Z=y)
  assert IF_map[1, 0] == pytest.approx(te, abs=1e-12)


@pytest.mark.parametrize("k", [1, 5])
def test_constant_condition_reproduces_unconditional(k):
  """With a constant Z every Z-distance is 0, so the CMI counts reduce to the MI counts."""
  rng = np.random.default_rng(k)
  n, tau, dt = 2000, 2, 0.5
  X = np.zeros((3, n))
  for t in range(n - 1):
    X[0, t + 1] = 0.5 * X[0, t] + rng.standard_normal()
    X[1, t + 1] = 0.5 * X[1, t] + 0.6 * X[0, t] + rng.standard_normal()
    X[2, t + 1] = 0.5 * X[2, t] + 0.6 * X[1, t] + rng.standard_normal()
  x, y = X[0].reshape(-1, 1), X[1].reshape(-1, 1)
  ref = information_flow(x, y, tau=tau, dt=dt, k=k)
  for z in (np.zeros((n, 1)), np.full((n, 2), 3.7)):
    assert information_flow(x, y, tau=tau, dt=dt, k=k, Z=z) == pytest.approx(ref, abs=1e-12)
  off = ~np.eye(3, dtype=bool)
  for mask in (None, off):
    ref_maps = information_flow_causal_map(X, tau, mask=mask, dt=dt, k=k, full=True)
    maps = information_flow_causal_map(X, tau, mask=mask, dt=dt, k=k, full=True, Z=np.zeros(n))
    for got, ref_map in zip(maps, ref_maps):
      np.testing.assert_allclose(got[off], ref_map[off], atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# 3. closed-form values (linear Gaussian VAR(1))
# ---------------------------------------------------------------------------

def test_exact_values_of_test_systems():
  """Sanity of the closed form: no direct X -> Y path gives zero conditional TE, and the
  Z = Y branch agrees with the transfer entropy formula of test_TE.py."""
  for A in (CHAIN, DRIVER):
    assert abs(_exact_cond_if(A, 1, [1, 2])) < 1e-12
  a, b, c = 0.5, 0.5, 0.7
  A2 = np.array([[a, 0.0, 0.0], [c, b, 0.0], [0.0, 0.0, 0.5]])
  S = solve_discrete_lyapunov(A2[:2, :2], np.eye(2))
  cov = b * S[1, 1] + c * S[0, 1]
  assert _exact_cond_if(A2, 1, [1]) == pytest.approx(0.5 * np.log(S[1, 1] - cov**2 / S[1, 1]), abs=1e-12)


# (system, tau, conditioning indices into s_t); [] = unconditional, [1] = TE, [1, 2] = conditional TE
ANALYTIC_CASES = [
  ("chain",  CHAIN,  1, []),      # 0.0378: indirect flow ...
  ("chain",  CHAIN,  1, [2]),     # -0.0001: ... removed by conditioning on the mediator
  ("chain",  CHAIN,  1, [1, 2]),  # 0 exactly
  ("chain",  CHAIN,  2, []),
  ("chain",  CHAIN,  2, [2]),
  ("chain",  CHAIN,  2, [1]),
  ("chain",  CHAIN,  2, [1, 2]),
  ("driver", DRIVER, 1, []),
  ("driver", DRIVER, 1, [2]),
  ("driver", DRIVER, 1, [1, 2]),  # 0 exactly
  ("direct", DIRECT, 1, [2]),
  ("direct", DIRECT, 1, [1, 2]),
]


@pytest.mark.parametrize("name, A, tau, cond", ANALYTIC_CASES,
                         ids=[f"{c[0]}-tau{c[2]}-Z{c[3]}" for c in ANALYTIC_CASES])
def test_conditional_information_flow_vs_analytical(name, A, tau, cond):
  # At N = 1e4 the mean bias over these cases is below 0.003 nats and the standard error of a
  # 16-trial mean is ~0.002 nats (N = 1e5: |bias| < 0.002 nats for all cases).
  n, trials, tol = 10000, 16, 0.01
  exact = _exact_cond_if(A, tau, cond)
  rng = np.random.default_rng(1000 * tau + len(cond))
  est = []
  for _ in range(trials):
    s = _simulate(A, n, rng)
    x, y = _c(s[:, :1]), _c(s[:, 1:2])
    z = _c(s[:, cond]) if cond else None
    est.append(information_flow(x, y, tau=tau, dt=1.0, k=5, Z=z))
  assert np.mean(est) == pytest.approx(exact, abs=tol), (
    f"{name} tau={tau} Z={cond}: est={np.mean(est):.4f}, exact={exact:.4f}")


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_conditional_input_validation():
  rng = np.random.default_rng(0)
  x, y = rng.standard_normal((200, 1)), rng.standard_normal((200, 1))
  with pytest.raises(RuntimeError):
    information_flow(x, y, Z=rng.standard_normal((199, 1)))   # wrong length
  with pytest.raises(RuntimeError):
    information_flow(x, y, Z=rng.standard_normal(200))        # must be 2D like X and Y
  with pytest.raises(ValueError):
    information_flow(x, y, Z=np.zeros((200, 0)))
  with pytest.raises(ValueError):
    information_flow(x, y, tau=-1)
  X = rng.standard_normal((3, 200))
  with pytest.raises(RuntimeError):
    information_flow_causal_map(X, 1, Z=rng.standard_normal(199))
  with pytest.raises(RuntimeError):
    information_flow_causal_map(X, 1, Z=rng.standard_normal((200, 2, 2)))
  with pytest.raises(ValueError):
    information_flow_causal_map(X, -1, Z=rng.standard_normal(200))
  assert np.all(np.isnan(information_flow_causal_map(X, 200, Z=rng.standard_normal(200))))


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

  # chain X -> W -> Y with coupling c on both links, tau = 2
  N, tau = 10000, 2
  cs = np.linspace(0.1e0, 0.9e0, 9)
  cs_fine = np.linspace(0.1e0, 0.9e0, 81)
  chain = lambda c: np.array([[0.5, 0.0, 0.0], [0.0, 0.3, c], [c, 0.0, 0.3]])
  est = {(): [], (2,): []}
  for c in cs:
    s = _simulate(chain(c), N, np.random.default_rng(42))
    x, y = _c(s[:, :1]), _c(s[:, 1:2])
    for cond in est:
      est[cond].append(information_flow(x, y, tau=tau, k=5, Z=_c(s[:, list(cond)]) if cond else None))

  plt.figure(figsize=(7, 6))
  styles = {(): ('blue', 'o', r'$T_{x \to y}$'), (2,): ('red', '^', r'$T_{x \to y | w}$')}
  for cond, (color, marker, label) in styles.items():
    plt.plot(cs_fine, [_exact_cond_if(chain(c), tau, list(cond)) for c in cs_fine],
             color=color, linestyle='solid')
    plt.plot(cs, est[cond], color=color, linestyle='none', marker=marker, label=label)
  plt.axhline(0.e0, color='gray', linestyle='dotted')
  plt.xlabel(r'$c$')
  plt.ylabel('Information flow')
  plt.legend(frameon=False)
  plt.savefig(_results / 'IF_conditional.png', dpi=150, bbox_inches='tight')
  plt.close()
  print(f'Saved {_results / "IF_conditional.png"}')


if __name__ == '__main__':
  save_results()
