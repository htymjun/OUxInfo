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
    IF_ref, _, _ = information_flow_causal_map(X, tau_arr, dt=1.0, k=5, n_threads=1, full=True)

    mask = np.ones((Nx, Nx), dtype=bool)
    IF_mask, _, _ = information_flow_causal_map(X, tau, mask=mask, dt=1.0, k=5, n_threads=1, full=True)
    for i in range(Nx):
        for j in range(Nx):
            if i == j:
                continue
            assert abs(IF_ref[i, j] - IF_mask[i, j]) < 1e-10, (
                f"[{i},{j}]: ref={IF_ref[i,j]:.8f}, mask={IF_mask[i,j]:.8f}")


def test_masked_pairs_are_zero():
    """Pairs absent from a symmetric mask must be 0.0 in the output."""
    rng = np.random.default_rng(7)
    Nx, Nt = 5, 200
    X = rng.standard_normal((Nx, Nt))

    # Symmetric sparse mask: only pairs (0,1) and (2,3) in both directions
    mask = np.zeros((Nx, Nx), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    mask[2, 3] = mask[3, 2] = True
    IF_mask, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, full=True)
    for i in range(Nx):
        for j in range(Nx):
            if i == j or mask[i, j]:
                continue
            assert IF_mask[i, j] == 0.0, f"[{i},{j}] should be 0 but got {IF_mask[i,j]}"


def test_causal_direction():
    """IF(x→y) > IF(y→x) for a unidirectionally driven AR system."""
    N = 5000
    a, b, c_xy, sigma = 0.5, 0.5, 0.5, 1.0
    rng = np.random.default_rng(99)
    x, y = _gaussian_ar_system(N, a, b, c_xy, sigma, rng)
    X = np.vstack([x, y])  # shape (2, N+1)

    mask = ~np.eye(2, dtype=bool)
    IF, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, full=True)
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
        IF, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, full=True)
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

    IF_1, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, n_threads=1, full=True)
    IF_4, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, n_threads=4, full=True)
    assert np.allclose(IF_1, IF_4, atol=1e-10, equal_nan=True), "Multi-thread result differs from single-thread"


# ---------------------------------------------------------------------------
# full=False tests
# ---------------------------------------------------------------------------

def test_full_false_returns_ndarray():
    """full=False must return an ndarray, not a tuple."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 200))
    result = information_flow_causal_map(X, full=False)
    assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
    assert result.shape == (4, 4)


def test_full_false_matches_full_true():
    """IF values from full=False must equal those from full=True (no mask)."""
    rng = np.random.default_rng(1)
    Nx, Nt = 5, 400
    X = rng.standard_normal((Nx, Nt))

    IF_full, _, _ = information_flow_causal_map(X, dt=1.0, k=5, full=True)
    IF_only = information_flow_causal_map(X, dt=1.0, k=5, full=False)

    np.testing.assert_allclose(IF_full, IF_only, rtol=1e-10,
                                err_msg="full=False IF differs from full=True IF")


def test_full_false_diagonal_is_nan():
    """Diagonal entries must be NaN for full=False."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 200))
    IF = information_flow_causal_map(X, full=False)
    for v in range(4):
        assert np.isnan(IF[v, v]), f"diagonal [{v},{v}] should be NaN"


def test_full_false_with_mask_matches_full_true():
    """Masked full=False IF values must equal masked full=True IF values."""
    rng = np.random.default_rng(3)
    Nx, Nt = 6, 400
    X = rng.standard_normal((Nx, Nt))
    mask = np.triu(np.ones((Nx, Nx), dtype=bool), k=1)

    IF_full, _, _ = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, full=True)
    IF_only = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, full=False)

    np.testing.assert_allclose(IF_full[mask], IF_only[mask], rtol=1e-10,
                                err_msg="Masked full=False IF differs from full=True IF")


def test_full_false_unmasked_entries_are_zero():
    """Pairs absent from a symmetric mask must remain 0.0 when full=False."""
    rng = np.random.default_rng(4)
    Nx, Nt = 5, 200
    X = rng.standard_normal((Nx, Nt))
    # Symmetric sparse mask: only pairs (0,1) and (2,3) in both directions
    mask = np.zeros((Nx, Nx), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    mask[2, 3] = mask[3, 2] = True

    IF = information_flow_causal_map(X, 1, mask=mask, dt=1.0, k=5, full=False)
    for i in range(Nx):
        for j in range(Nx):
            if i == j or mask[i, j]:
                continue
            assert IF[i, j] == 0.0, f"[{i},{j}] should be 0 but got {IF[i,j]}"


def test_full_false_n_threads_same_result():
    """full=False: single-threaded and multi-threaded results must be identical."""
    rng = np.random.default_rng(5)
    Nx, Nt = 4, 300
    X = rng.standard_normal((Nx, Nt))

    IF_1 = information_flow_causal_map(X, 1, dt=1.0, k=5, n_threads=1, full=False)
    IF_4 = information_flow_causal_map(X, 1, dt=1.0, k=5, n_threads=4, full=False)
    assert np.allclose(IF_1, IF_4, atol=1e-10, equal_nan=True), \
        "full=False multi-thread result differs from single-thread"


