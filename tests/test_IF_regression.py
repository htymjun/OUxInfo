import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov
from ouxinfo import information_flow, mutual_info


N_VALUES = [1000, 2000, 5000, 10000]
K = 5


def _time_call(fn, *args, **kwargs):
    fn(*args, **kwargs)  # warmup
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def _if_reference(x, y, tau, dt, k):
    """Old two-call approach: builds X marginal twice."""
    Neff = len(x) - tau
    Ilag = mutual_info(x[:Neff], y[tau:tau+Neff], k=k)
    I    = mutual_info(x[:Neff], y[:Neff],        k=k)
    return (Ilag - I) / dt


def _analytical_if_gaussian_ar(a, b, c_xy, tau=1, dt=1.0, sigma=1.0):
    A = np.array([[a, 0.0], [c_xy, b]])
    Q = np.array([[sigma**2, 0.0], [0.0, sigma**2]])
    Sigma = solve_discrete_lyapunov(A, Q)
    S_xx, S_yy = Sigma[0, 0], Sigma[1, 1]
    Gamma_tau = np.linalg.matrix_power(A, tau) @ Sigma
    rho_0   = Sigma[1, 0]     / np.sqrt(S_xx * S_yy)
    rho_tau = Gamma_tau[1, 0] / np.sqrt(S_xx * S_yy)
    I_0   = -0.5 * np.log(1.0 - rho_0**2)
    I_tau = -0.5 * np.log(1.0 - rho_tau**2)
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


def test_information_flow_vs_reference():
    """New C++ result must be numerically identical to the two-call reference."""
    rng = np.random.default_rng(42)
    for N in [500, 2000]:
        x = rng.standard_normal((N, 1))
        y = rng.standard_normal((N, 1))
        new_val = information_flow(x, y, tau=1, dt=1.0, k=K)
        ref_val = _if_reference(x, y, tau=1, dt=1.0, k=K)
        assert abs(new_val - ref_val) < 1e-10, (
            f"N={N}: new={new_val:.8f}, ref={ref_val:.8f}, diff={abs(new_val-ref_val):.2e}")


def test_information_flow_vs_analytical():
    """New implementation must match the Gaussian AR analytical value within 10%."""
    nt = 500
    n_trials = 50
    a, b, sigma = 0.5, 0.5, 1.0
    for c_xy in [0.3, 0.5, 0.7]:
        IF_true = _analytical_if_gaussian_ar(a, b, c_xy, tau=1, dt=1.0, sigma=sigma)
        rng = np.random.default_rng(int(c_xy * 100))
        IF_sum = 0.0
        for _ in range(n_trials):
            x, y = _gaussian_ar_system(nt, a, b, c_xy, sigma, rng)
            IF_sum += information_flow(x.reshape(-1, 1), y.reshape(-1, 1), tau=1, dt=1.0, k=K)
        IF_est = IF_sum / n_trials
        tol = 0.1 * IF_true
        assert np.isclose(IF_est, IF_true, atol=tol), (
            f"c_xy={c_xy}: est={IF_est:.4f}, true={IF_true:.4f}, tol={tol:.4f}")


def test_information_flow_speed():
    """New C++ path (X marginal built once) must beat the two-call reference."""
    rng = np.random.default_rng(0)
    times_new, times_ref = [], []

    for N in N_VALUES:
        x = rng.standard_normal((N, 1))
        y = rng.standard_normal((N, 1))
        times_new.append(_time_call(information_flow, x, y, tau=1, dt=1.0, k=K))
        times_ref.append(_time_call(_if_reference, x, y, tau=1, dt=1.0, k=K))

    print("\nInformation flow speed comparison:")
    print(f"{'N':>8}  {'new (s)':>10}  {'ref (s)':>10}  {'speedup':>8}")
    for N, tn, tr in zip(N_VALUES, times_new, times_ref):
        print(f"{N:>8}  {tn:>10.4f}  {tr:>10.4f}  {tr/tn:>7.2f}x")

    assert times_new[-1] < times_ref[-1], (
        f"new ({times_new[-1]:.4f}s) not faster than ref ({times_ref[-1]:.4f}s) "
        f"at N={N_VALUES[-1]}")

    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'xtick.direction': 'in', 'ytick.direction': 'in', 'font.size': 16,
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_VALUES, times_new, 'o-', label='information_flow (new)')
    ax.loglog(N_VALUES, times_ref, 's-', label='reference (2× mutual_info)')
    ax.set_xlabel(r'$N$')
    ax.set_ylabel('Time (s)')
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'speed_IF.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
