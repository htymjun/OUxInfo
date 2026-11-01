"""Consistency and argument checks of the Python-level helpers and the C++ wrappers."""
import numpy as np
import pytest
from ouxinfo import (auto_mutual_info, backward_transfer_entropy, transfer_entropy,
                     transfer_entropy_causal_map, information_flow, information_flow_causal_map,
                     KL_div, shannon_entropy, mutual_info, conditional_mutual_info,
                     entropy_rate, TEIFL)


def test_auto_mutual_info_returns_lag_of_first_minimum():
  rng = np.random.default_rng(0)
  t = np.arange(20000)
  x = np.sin(2 * np.pi * t / 40) + 0.3 * rng.standard_normal(t.size)
  taus, mi, opt = auto_mutual_info(x, 20, k=5)
  assert taus[0] == 1
  i = int(np.flatnonzero(taus == opt)[0])
  assert mi[i] < mi[i - 1] and mi[i] < mi[i + 1]
  assert not any(mi[j] < mi[j - 1] and mi[j] < mi[j + 1] for j in range(1, i))


def test_backward_transfer_entropy_uses_trial_argument():
  rng = np.random.default_rng(1)
  x = rng.standard_normal((2000, 1))
  y = np.roll(x, 1, axis=0) + 0.5 * rng.standard_normal((2000, 1))
  ref = transfer_entropy(np.ascontiguousarray(x[::-1]), np.ascontiguousarray(y[::-1]), m=2, trial=0)
  assert backward_transfer_entropy(x, y, m=2, trial=0) == pytest.approx(ref, abs=1e-12)


def test_teifl_embedding_map_matches_mte_and_bte(tmp_path):
  rng = np.random.default_rng(2)
  n = 600
  X = np.zeros((3, n))
  for t in range(n - 1):
    X[0, t + 1] = 0.8 * X[0, t] + rng.standard_normal()
    X[1, t + 1] = 0.7 * X[1, t] + 0.6 * X[0, t] + rng.standard_normal()
    X[2, t + 1] = 0.9 * X[2, t] + rng.standard_normal()
  TEIFL(X, max_m=4, tol=0.05, k=5, result_dir=str(tmp_path), multi=True)
  r = np.load(tmp_path / 'teifl.npz')
  for j in range(3):
    for i in range(3):
      if i == j:
        continue
      m = int(r['mmap'][j, i])
      assert 2 <= m <= 4
      xi, xj = X[i].reshape(-1, 1), X[j].reshape(-1, 1)
      assert r['mTE'][j, i] == pytest.approx(transfer_entropy(xi, xj, m=m, k=5), abs=1e-12)
      assert r['bTE'][j, i] == pytest.approx(backward_transfer_entropy(xi, xj, m=m, k=5), abs=1e-12)


def test_invalid_arguments_raise():
  rng = np.random.default_rng(3)
  x = rng.standard_normal((300, 1))
  X = rng.standard_normal((2, 100))
  tau2 = np.ones(2, dtype=np.int32)
  for call in (lambda: transfer_entropy(x, x, m=0), lambda: transfer_entropy(x, x, m=2, lag=0),
               lambda: transfer_entropy(x, x, tau=-1), lambda: transfer_entropy(x, x, tau=0),
               lambda: information_flow(x, x, tau=0), lambda: information_flow_causal_map(X, 0),
               lambda: transfer_entropy_causal_map(X, 0 * tau2),
               lambda: transfer_entropy_causal_map(X, -tau2), lambda: transfer_entropy_causal_map(X, tau2, m=0),
               lambda: entropy_rate(x, tau=0),
               lambda: TEIFL(X, max_m=1, multi=True)):
    with pytest.raises(ValueError):
      call()
  with pytest.raises(RuntimeError):
    KL_div(rng.standard_normal((100, 2)), rng.standard_normal((100, 1)))
  for Z in (None, rng.standard_normal(100)):
    with pytest.raises(RuntimeError):
      information_flow_causal_map(rng.standard_normal((3, 100)), np.ones(1, dtype=np.int32), Z=Z)


# ---------------------------------------------------------------------------
# inputs that used to crash the interpreter or return silently wrong values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(bad):
  """NaN made std::sort read out of bounds (segfault); inf made a self-count of -1 and
  digamma(0) threw inside an OpenMP region (std::terminate)."""
  rng = np.random.default_rng(4)
  x, y, z = (rng.standard_normal((500, 1)) for _ in range(3))
  b = x.copy(); b[7, 0] = bad
  calls = [lambda: shannon_entropy(b), lambda: KL_div(b, y), lambda: KL_div(x, b),
           lambda: mutual_info(b, y), lambda: mutual_info(x, b, Thei=2),
           lambda: conditional_mutual_info(b, y, z), lambda: conditional_mutual_info(x, y, b),
           lambda: transfer_entropy(b, y), lambda: transfer_entropy(x, b),
           lambda: information_flow(b, y), lambda: information_flow(x, b),
           lambda: information_flow(x, y, Z=b)]
  for call in calls:
    with pytest.raises(ValueError):
      call()
  X = rng.standard_normal((3, 300)); X[1, 5] = bad
  for call in (lambda: information_flow_causal_map(X, 1), lambda: information_flow_causal_map(X, 1, full=True, mask=~np.eye(3, dtype=bool)),
               lambda: transfer_entropy_causal_map(X, np.ones(3, dtype=np.int32))):
    with pytest.raises(ValueError):
      call()
  Z = rng.standard_normal(300); Z[3] = bad
  with pytest.raises(ValueError):
    information_flow_causal_map(rng.standard_normal((3, 300)), 1, Z=Z)


def test_zero_columns_raise():
  y = np.random.default_rng(5).standard_normal((100, 1))
  e = np.zeros((100, 0))
  for call in (lambda: shannon_entropy(e), lambda: KL_div(e, e), lambda: mutual_info(e, y),
               lambda: conditional_mutual_info(y, y, e), lambda: transfer_entropy(y, e),
               lambda: information_flow(y, e), lambda: information_flow(y, y, Z=e),
               lambda: information_flow_causal_map(np.zeros((3, 100, 0))),
               lambda: transfer_entropy_causal_map(np.zeros((3, 100, 0)), np.ones(3, dtype=np.int32)),
               lambda: information_flow_causal_map(np.zeros((0, 100)), np.zeros(0, dtype=np.int32))):
    with pytest.raises(ValueError):
      call()


def test_causal_map_restores_thread_count():
  import ctypes, ctypes.util
  lib = ctypes.util.find_library('gomp')
  if lib is None:
    pytest.skip("libgomp not found")
  gomp = ctypes.CDLL(lib)
  before = gomp.omp_get_max_threads()
  X = np.random.default_rng(6).standard_normal((3, 300))
  information_flow_causal_map(X, 1, n_threads=1 if before > 1 else 2)
  transfer_entropy_causal_map(X, np.ones(3, dtype=np.int32), n_threads=1 if before > 1 else 2)
  assert gomp.omp_get_max_threads() == before
  with pytest.raises(ValueError):
    information_flow_causal_map(X, 1, n_threads=0)


def test_huge_theiler_window_gives_nan():
  x = np.random.default_rng(7).standard_normal((500, 1))
  assert np.isnan(mutual_info(x, x[::-1].copy(), Thei=2**30))   # k + 2 Thei + 1 overflowed int


def test_surrogates_are_reproducible():
  rng = np.random.default_rng(8)
  X = rng.standard_normal((4, 600))
  tau = np.ones(4, dtype=np.int32)
  a = transfer_entropy_causal_map(X, tau, trial=3, n_threads=1)
  b = transfer_entropy_causal_map(X, tau, trial=3, n_threads=4)
  np.testing.assert_allclose(a, b, atol=1e-12)                    # used to depend on thread scheduling
  x, y = X[0].reshape(-1, 1), X[1].reshape(-1, 1)
  assert transfer_entropy(x, y, trial=3, seed=7) == pytest.approx(transfer_entropy(x, y, trial=3, seed=7), abs=1e-12)


def test_duplicates_and_unclipped_kl():
  c = np.ones((100, 1))
  for v in (shannon_entropy(c), mutual_info(c, c), conditional_mutual_info(c, c, c), KL_div(c, c)):
    assert np.isnan(v)                                            # was -inf, 0.0, 0.0, 0.0
  rng = np.random.default_rng(9)
  kl = [KL_div(rng.standard_normal((500, 1)), rng.standard_normal((500, 1))) for _ in range(20)]
  assert min(kl) < 0.0                                            # was clipped to 0, biasing the mean
  assert abs(np.mean(kl)) < 0.05


def test_rates_are_divided_by_tau_dt():
  """Every rate is per unit time: Delta / (tau * dt), with dt the sampling step."""
  rng = np.random.default_rng(10)
  n, tau, dt = 3000, 3, 0.2
  X = np.zeros((2, n))
  for t in range(n - 1):
    X[0, t + 1] = 0.8 * X[0, t] + rng.standard_normal()
    X[1, t + 1] = 0.5 * X[1, t] + 0.6 * X[0, t] + rng.standard_normal()
  x, y = X[0].reshape(-1, 1), X[1].reshape(-1, 1)
  m = n - tau
  dI = mutual_info(x[:m], y[tau:]) - mutual_info(x[:m], y[:m])
  assert information_flow(x, y, tau=tau, dt=dt) == pytest.approx(dI / (tau * dt), abs=1e-12)
  assert information_flow_causal_map(X, tau, dt=dt)[1, 0] == pytest.approx(dI / (tau * dt), abs=1e-12)
  te = conditional_mutual_info(x[:m], np.ascontiguousarray(y[tau:]), y[:m])
  assert transfer_entropy(x, y, tau=tau, dt=dt) == pytest.approx(te / (tau * dt), abs=1e-12)
  assert transfer_entropy_causal_map(X, np.full(2, tau, dtype=np.int32), dt=dt)[1, 0] == pytest.approx(te / (tau * dt), abs=1e-12)
  dH = shannon_entropy(x[tau:]) - shannon_entropy(x[:-tau])
  assert entropy_rate(x, tau=tau, dt=dt) == pytest.approx(dH / (tau * dt), abs=1e-12)
