"""Exactness checks of the k-NN machinery against brute-force references.

The estimators are deterministic functions of the data, so on identical inputs they must agree
with an exhaustive-distance implementation of the same formula up to rounding. These tests guard
against (i) kd-tree pruning that is inconsistent with the Chebyshev metric, (ii) boundary rounding
in the 1-D sorted-array counts, (iii) dropping points with zero marginal counts (k = 1), and
(iv) a wrong unit-ball constant in the Kozachenko-Leonenko entropy.
"""
import numpy as np
import pytest
from scipy.special import digamma
from ouxinfo import (shannon_entropy, KL_div, mutual_info, conditional_mutual_info,
                     information_flow, information_flow_causal_map, transfer_entropy,
                     transfer_entropy_causal_map)


def _cheb(a, b):
  return np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=-1)


def _ksg1(x, y, k):
  N = len(x); dx, dy = _cheb(x, x), _cheb(y, y)
  eps = np.partition(np.maximum(dx, dy), k, axis=1)[:, k][:, None]
  nx = (dx < eps).sum(1) - 1; ny = (dy < eps).sum(1) - 1
  return digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))


def _cmi(x, y, z, k):
  dx, dy, dz = _cheb(x, x), _cheb(y, y), _cheb(z, z)
  eps = np.partition(np.maximum(np.maximum(dx, dy), dz), k, axis=1)[:, k][:, None]
  nz = (dz < eps).sum(1) - 1
  nxz = (np.maximum(dx, dz) < eps).sum(1) - 1; nyz = (np.maximum(dy, dz) < eps).sum(1) - 1
  return digamma(k) + np.mean(digamma(nz + 1) - digamma(nxz + 1) - digamma(nyz + 1))


def _entropy(x, k):
  N, d = x.shape
  r = np.partition(_cheb(x, x), k, axis=1)[:, k]
  return -digamma(k) + digamma(N) + d * np.mean(np.log(2 * r))


def _kl(x, y, k):
  n, d = x.shape
  rho = np.partition(_cheb(x, x), k, axis=1)[:, k]
  nu = np.partition(_cheb(x, y), k - 1, axis=1)[:, k - 1]
  return max(d * np.mean(np.log(nu / rho)) + np.log(len(y) / (n - 1)), 0.0)


def _corr_gauss(n, d, r, rng):
  C = np.full((d, d), r) + (1 - r) * np.eye(d)
  return rng.multivariate_normal(np.zeros(d), C, n)


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("dims", [(1, 1), (1, 2), (2, 2), (3, 3)])
def test_mutual_info_matches_brute_force(k, dims):
  rng = np.random.default_rng(10 * k + dims[0] + dims[1])
  dx, dy = dims
  Z = _corr_gauss(1500, dx + dy, 0.6, rng)
  x, y = np.ascontiguousarray(Z[:, :dx]), np.ascontiguousarray(Z[:, dx:])
  assert mutual_info(x, y, k=k) == pytest.approx(_ksg1(x, y, k), abs=1e-12)


@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("dz", [1, 2])
def test_conditional_mutual_info_matches_brute_force(k, dz):
  rng = np.random.default_rng(k + 7 * dz)
  Z = _corr_gauss(1200, 2 + dz, 0.5, rng)
  x, y, z = (np.ascontiguousarray(a) for a in (Z[:, :1], Z[:, 1:2], Z[:, 2:]))
  assert conditional_mutual_info(x, y, z, k=k) == pytest.approx(_cmi(x, y, z, k), abs=1e-12)


@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("d", [1, 2, 4])
def test_entropy_and_kl_match_brute_force(k, d):
  rng = np.random.default_rng(k + 3 * d)
  x = rng.standard_normal((1500, d)); y = 1.5 * rng.standard_normal((1500, d)) + 0.3
  assert shannon_entropy(x, k=k) == pytest.approx(_entropy(x, k), abs=1e-12)
  assert KL_div(x, y, k=k) == pytest.approx(_kl(x, y, k), abs=1e-12)


@pytest.mark.parametrize("d, tol", [(2, 0.012), (3, 0.025)])
def test_entropy_gaussian_multidimensional(d, tol):
  # the pre-fix unit-ball constant was off by log(pi/4) = -0.24 (d=2) and -0.65 (d=3)
  rng = np.random.default_rng(d)
  H = [shannon_entropy(rng.standard_normal((20000, d)), k=5) for _ in range(5)]
  H_true = 0.5 * d * np.log(2 * np.pi * np.e)
  assert np.mean(H) == pytest.approx(H_true, abs=tol)


def test_mutual_info_high_dimensional_gaussian():
  """I(X1; (X2..X6)) for equicorrelated Gaussians (Kraskov et al. 2004, Fig. 14 setting)."""
  rng = np.random.default_rng(3); r, m = 0.9, 6
  C = np.full((m, m), r) + (1 - r) * np.eye(m)
  I_true = 0.5 * np.log(C[0, 0] * np.linalg.det(C[1:, 1:]) / np.linalg.det(C))
  Z = rng.multivariate_normal(np.zeros(m), C, 20000)
  I_est = mutual_info(np.ascontiguousarray(Z[:, :1]), np.ascontiguousarray(Z[:, 1:]), k=3)
  assert I_est == pytest.approx(I_true, abs=0.03)


def test_information_flow_and_maps_match_brute_force():
  rng = np.random.default_rng(5); n, tau, dt, k = 1000, 2, 0.1, 3
  X = np.zeros((3, n))
  for t in range(n - 1):
    X[0, t + 1] = 0.5 * X[0, t] + rng.standard_normal()
    X[1, t + 1] = 0.5 * X[1, t] + 0.6 * X[0, t] + rng.standard_normal()
    X[2, t + 1] = 0.5 * X[2, t] + 0.6 * X[1, t] + rng.standard_normal()
  m = n - tau
  ref = np.full((3, 3), np.nan)
  for i in range(3):
    for j in range(3):
      if i != j:
        xi, xj = X[i].reshape(-1, 1), X[j].reshape(-1, 1)
        ref[j, i] = (_ksg1(xi[:m], xj[tau:], k) - _ksg1(xi[:m], xj[:m], k)) / dt
  off = ~np.eye(3, dtype=bool)
  x0, x1 = X[0].reshape(-1, 1), X[1].reshape(-1, 1)
  assert information_flow(x0, x1, tau=tau, dt=dt, k=k) == pytest.approx(ref[1, 0], abs=1e-10)
  for full in (False, True):
    for mask in (None, off):
      out = information_flow_causal_map(np.ascontiguousarray(X), tau, mask=mask, dt=dt, k=k, full=full)
      IF = out[0] if full else out
      np.testing.assert_allclose(IF[off], ref[off], atol=1e-10)


@pytest.mark.parametrize("k", [1, 5])
def test_too_few_samples_give_nan(k):
  """With N <= k there is no k-th neighbour; the estimators used to return 0 (MI family),
  -inf (entropy) or a stale value (KL) instead of NaN. N = k + 1 is the smallest valid size."""
  rng = np.random.default_rng(k)
  sample = lambda n: rng.standard_normal((n, 1))
  for n, invalid in ((k, True), (k + 1, False)):
    x, y, z = sample(n), sample(n), sample(n)
    vals = [shannon_entropy(x, k=k), KL_div(x, sample(n), k=k), mutual_info(x, y, k=k),
            conditional_mutual_info(x, y, z, k=k)]
    assert np.all(np.isnan(vals)) if invalid else np.all(np.isfinite(vals)), (n, vals)

  # KL also needs k points in Y
  assert np.isnan(KL_div(sample(50), sample(k - 1), k=k))
  assert np.isfinite(KL_div(sample(50), sample(k), k=k))

  # Theiler window: the middle point has N - (2 Thei + 1) admissible neighbours
  thei = 3
  n0 = k + 2 * thei + 1
  assert np.isnan(mutual_info(sample(n0 - 1), sample(n0 - 1), k=k, Thei=thei))
  assert np.isfinite(mutual_info(sample(n0), sample(n0), k=k, Thei=thei))

  # time-lagged estimators: N - tau usable samples (also covers N <= tau)
  tau, off = 2, ~np.eye(3, dtype=bool)
  for n, invalid in ((1, True), (k + tau, True), (k + tau + 1, False)):
    x, y, z = sample(n), sample(n), sample(n)
    X = rng.standard_normal((3, n))
    vals = [information_flow(x, y, tau=tau, k=k), information_flow(x, y, tau=tau, k=k, Z=z),
            transfer_entropy_causal_map(X, np.full(3, tau, dtype=np.int32), k=k)[off]]
    if n > tau:
      vals.append(transfer_entropy(x, y, tau=tau, k=k))
    for full in (False, True):
      for mask in (None, off):
        out = information_flow_causal_map(X, tau, mask=mask, k=k, full=full)
        vals += [m[off] for m in (out if full else (out,))]
    vals = np.concatenate([np.ravel(v) for v in vals])
    assert np.all(np.isnan(vals)) if invalid else np.all(np.isfinite(vals)), (n, vals)


def test_invalid_k_and_theiler_window_raise():
  x = np.random.default_rng(0).standard_normal((100, 1))
  X = np.ascontiguousarray(x.reshape(2, 50))
  for call in (lambda: shannon_entropy(x, k=0), lambda: KL_div(x, x, k=0),
               lambda: mutual_info(x, x, k=0), lambda: conditional_mutual_info(x, x, x, k=0),
               lambda: transfer_entropy(x, x, k=0), lambda: information_flow(x, x, k=0),
               lambda: transfer_entropy_causal_map(X, np.ones(2, dtype=np.int32), k=0),
               lambda: information_flow_causal_map(X, 1, k=0),
               lambda: mutual_info(x, x, Thei=-1)):
    with pytest.raises(ValueError):
      call()


def test_causal_map_rejects_per_variable_tau():
  X = np.random.default_rng(0).standard_normal((3, 300))
  with pytest.raises(ValueError):
    information_flow_causal_map(X, np.array([1, 2, 3], dtype=np.int32))
