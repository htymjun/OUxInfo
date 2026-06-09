import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from ouxinfo import information_flow_causal_map


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gaussian_ar_system(nt, a, b, c_xy, sigma, rng):
    x = np.zeros(nt + 1)
    y = np.zeros(nt + 1)
    x[0] = rng.normal(0.0, sigma)
    y[0] = rng.normal(0.0, sigma)
    for i in range(nt):
        x[i + 1] = a * x[i] + rng.normal(0.0, sigma)
        y[i + 1] = b * y[i] + c_xy * x[i] + rng.normal(0.0, sigma)
    return x, y


def _analytical_if_gaussian_ar(a, b, c_xy, tau=1, dt=1.0, sigma=1.0):
    A = np.array([[a, 0.0], [c_xy, b]])
    Q = np.array([[sigma**2, 0.0], [0.0, sigma**2]])
    Sigma = solve_discrete_lyapunov(A, Q)
    S_xx, S_yy = Sigma[0, 0], Sigma[1, 1]
    Gamma_tau = np.linalg.matrix_power(A, tau) @ Sigma
    rho_0   = Sigma[1, 0]     / np.sqrt(S_xx * S_yy)
    rho_tau = Gamma_tau[1, 0] / np.sqrt(S_xx * S_yy)
    return (-0.5 * np.log(1.0 - rho_tau**2) - (-0.5 * np.log(1.0 - rho_0**2))) / dt


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_full_mask_matches_information_flow_causal_map():
    """All-True mask must give the same result as the no-mask path."""
    rng = np.random.default_rng(42)
    Nx, Nt = 6, 400
    X = rng.standard_normal((Nx, Nt))
    tau = 1

    tau_arr = np.full(Nx, tau, dtype=np.int32)
    IF_ref, _, _ = information_flow_causal_map(X, tau_arr, dt=1.0, k=5, n_threads=1)

    mask = np.ones((Nx, Nx), dtype=bool)
    IF_mask, _, _ = information_flow_causal_map(X, tau, mask=mask, dt=1.0, k=5, n_threads=1)
    for i in range(Nx):
        for j in range(Nx):
            if i == j:
                continue
            assert abs(IF_ref[i, j] - IF_mask[i, j]) < 1e-10, (
                f"[{i},{j}]: ref={IF_ref[i,j]:.8f}, mask={IF_mask[i,j]:.8f}")


def test_masked_pairs_are_zero():
    """Entries where mask is False must be 0.0 in the output."""
    rng = np.random.default_rng(7)
    Nx, Nt = 5, 200
    X = rng.standard_normal((Nx, Nt))

    # Only upper-triangle entries
    mask = np.triu(np.ones((Nx, Nx), dtype=bool), k=1)
    IF_mask, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5)
    for i in range(Nx):
        for j in range(Nx):
            if i == j or mask[i, j]:
                continue  # diagonal and True entries are allowed to be non-zero
            assert IF_mask[i, j] == 0.0, f"[{i},{j}] should be 0 but got {IF_mask[i,j]}"


def test_causal_direction():
    """IF(x→y) > IF(y→x) for a unidirectionally driven AR system."""
    N = 5000
    a, b, c_xy, sigma = 0.5, 0.5, 0.5, 1.0
    rng = np.random.default_rng(99)
    x, y = _gaussian_ar_system(N, a, b, c_xy, sigma, rng)
    X = np.vstack([x, y])  # shape (2, N+1)

    mask = ~np.eye(2, dtype=bool)
    IF, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5)
    # IF[i,j] = IF from X_j to X_i
    # X_0 = x, X_1 = y;  x drives y → IF[1,0] = IF(x→y) > IF[0,1] = IF(y→x)
    assert IF[1, 0] > IF[0, 1], (
        f"Expected IF(x→y)={IF[1,0]:.4f} > IF(y→x)={IF[0,1]:.4f}")


def test_vs_analytical():
    """Averaged estimate must be within 10 % of the Gaussian AR analytical value."""
    nt, n_trials = 500, 50
    a, b, c_xy, sigma = 0.5, 0.5, 0.5, 1.0
    IF_true = _analytical_if_gaussian_ar(a, b, c_xy, tau=1, dt=1.0, sigma=sigma)

    rng = np.random.default_rng(55)
    IF_sum = 0.0
    mask = ~np.eye(2, dtype=bool)
    for _ in range(n_trials):
        x, y = _gaussian_ar_system(nt, a, b, c_xy, sigma, rng)
        X = np.vstack([x, y])
        IF, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5)
        IF_sum += IF[1, 0]  # IF(x→y)
    IF_est = IF_sum / n_trials

    tol = 0.1 * IF_true
    assert np.isclose(IF_est, IF_true, atol=tol), (
        f"est={IF_est:.4f}, true={IF_true:.4f}, tol={tol:.4f}")


def test_n_threads_same_result():
    """Single-threaded and multi-threaded runs must produce identical output."""
    rng = np.random.default_rng(11)
    Nx, Nt = 4, 300
    X = rng.standard_normal((Nx, Nt))
    mask = ~np.eye(Nx, dtype=bool)

    IF_1, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, n_threads=1)
    IF_4, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, n_threads=4)
    assert np.allclose(IF_1, IF_4, atol=1e-10, equal_nan=True), "Multi-thread result differs from single-thread"


