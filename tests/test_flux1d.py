import numpy as np
from ouxinfo import information_flow_1d


def _make_driven_chain(nx, nt, alpha, beta, sigma, rng):
    """1D AR chain where cell i drives cell i+1 unidirectionally.

    x[i, t+1] = alpha*x[i, t] + sigma*eps
    x[i+1, t+1] = alpha*x[i+1, t] + beta*x[i, t] + sigma*eps
    """
    data = np.zeros((nx, nt + 1))
    data[:, 0] = rng.normal(0.0, sigma, nx)
    for t in range(nt):
        eps = rng.normal(0.0, sigma, nx)
        data[0, t + 1] = alpha * data[0, t] + eps[0]
        for i in range(1, nx):
            data[i, t + 1] = alpha * data[i, t] + beta * data[i - 1, t] + eps[i]
    return data[:, 1:]  # drop initial condition → shape (nx, nt)


def test_flux1d_output_shape():
    rng = np.random.default_rng(42)
    nx, nt = 5, 3000
    data = rng.standard_normal((nx, nt))
    result = information_flow_1d(data, dt=1.0, tau=1, k=5)
    for key in ('J_fwd', 'J_bwd', 'J_net', 'J_sym', 'Leak'):
        assert key in result, f"missing key '{key}'"
        assert result[key].shape == (nx - 1,), (
            f"{key}: expected shape ({nx-1},), got {result[key].shape}")


def test_flux1d_derived_quantities():
    rng = np.random.default_rng(42)
    nx, nt = 3, 3000
    data = rng.standard_normal((nx, nt))
    r = information_flow_1d(data, dt=1.0, tau=1, k=5)
    np.testing.assert_allclose(r['J_net'], r['J_fwd'] - r['J_bwd'])
    np.testing.assert_allclose(r['J_sym'], 0.5 * (r['J_fwd'] + r['J_bwd']))


def test_flux1d_causal_direction():
    nx, nt = 3, 5000
    alpha, beta, sigma = 0.5, 0.5, 1.0
    rng = np.random.default_rng(42)
    data = _make_driven_chain(nx, nt, alpha, beta, sigma, rng)
    r = information_flow_1d(data, dt=1.0, tau=1, k=5)
    for i in range(nx - 1):
        assert r['J_net'][i] > 0, (
            f"interface {i}: J_net={r['J_net'][i]:.4f} should be positive "
            f"(J_fwd={r['J_fwd'][i]:.4f}, J_bwd={r['J_bwd'][i]:.4f})")


def test_flux1d_independent_cells_small_net_flux():
    nx, nt = 3, 5000
    rng = np.random.default_rng(42)
    # Fully independent AR(1) series — no coupling
    alpha, sigma = 0.5, 1.0
    data = np.zeros((nx, nt))
    data[:, 0] = rng.normal(0.0, sigma, nx)
    for t in range(1, nt):
        data[:, t] = alpha * data[:, t - 1] + rng.normal(0.0, sigma, nx)
    r = information_flow_1d(data, dt=1.0, tau=1, k=5)
    for i in range(nx - 1):
        assert abs(r['J_net'][i]) < 0.05, (
            f"interface {i}: |J_net|={abs(r['J_net'][i]):.4f} should be < 0.05 for independent cells")


def test_flux1d_input_validation():
    import pytest
    with pytest.raises(ValueError, match="2D"):
        information_flow_1d(np.zeros((2, 3, 4, 5)))
    with pytest.raises(ValueError, match="nx >= 2"):
        information_flow_1d(np.zeros((1, 100)))
    with pytest.raises(ValueError, match="nx >= 2"):
        information_flow_1d(np.zeros((4, 1, 100)))


def test_flux1d_3d_output_shape():
    rng = np.random.default_rng(42)
    nz, nx, nt = 3, 5, 1000
    data = rng.standard_normal((nz, nx, nt))
    result = information_flow_1d(data, dt=1.0, tau=1, k=5)
    for key in ('J_fwd', 'J_bwd', 'J_net', 'J_sym', 'Leak'):
        assert key in result, f"missing key '{key}'"
        assert result[key].shape == (nx - 1,), (
            f"{key}: expected shape ({nx-1},), got {result[key].shape}")


def test_flux1d_3d_derived_quantities():
    rng = np.random.default_rng(42)
    nz, nx, nt = 2, 3, 1000
    data = rng.standard_normal((nz, nx, nt))
    r = information_flow_1d(data, dt=1.0, tau=1, k=5)
    np.testing.assert_allclose(r['J_net'], r['J_fwd'] - r['J_bwd'])
    np.testing.assert_allclose(r['J_sym'], 0.5 * (r['J_fwd'] + r['J_bwd']))


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

    nx, nt = 6, 3000
    alpha, beta, sigma = 0.5, 0.5, 1.0
    rng = np.random.default_rng(42)
    data = _make_driven_chain(nx, nt, alpha, beta, sigma, rng)
    result = information_flow_1d(data, dt=1.0, tau=1, k=5)
    x_iface = np.arange(nx - 1) + 0.5

    # plot 1: spatiotemporal map
    plt.figure(figsize=(8, 4))
    plt.imshow(data[:, :200], aspect='auto', origin='lower', cmap='RdBu_r',
               extent=[0, 200, -0.5, nx - 0.5])
    plt.xlabel(r'$t$', fontsize=20, style='italic')
    plt.ylabel(r'$i$', fontsize=20, style='italic')
    plt.colorbar(label=r'$q_i(t)$')
    plt.savefig(_results / 'flux1d_map.png', dpi=150, bbox_inches='tight')
    plt.close()

    # plot 2: directional fluxes
    plt.figure(figsize=(7, 7))
    plt.plot(x_iface, result['J_fwd'], color='blue',  linestyle='solid',  label=r'$J_{i \to i+1}$')
    plt.plot(x_iface, result['J_bwd'], color='red',   linestyle='solid',  label=r'$J_{i+1 \to i}$')
    plt.plot(x_iface, result['J_net'], color='black', linestyle='dashed', label=r'$J^{\rm net}$')
    plt.axhline(0.0, color='gray', linestyle='dotted')
    plt.xlabel(r'$i + \frac{1}{2}$', fontsize=20)
    plt.ylabel(r'information flux', fontsize=20)
    plt.legend(frameon=False)
    plt.savefig(_results / 'flux1d_J.png', dpi=150, bbox_inches='tight')
    plt.close()

    # plot 3: coupling strength vs leak
    plt.figure(figsize=(7, 7))
    plt.plot(x_iface, result['J_sym'], color='blue', linestyle='solid',  label=r'$J^{\rm sym}$')
    plt.plot(x_iface, result['Leak'],  color='red',  linestyle='dashed', label=r'Leak')
    plt.axhline(0.0, color='gray', linestyle='dotted')
    plt.xlabel(r'$i + \frac{1}{2}$', fontsize=20)
    plt.ylabel(r'information flux', fontsize=20)
    plt.legend(frameon=False)
    plt.savefig(_results / 'flux1d_Jsym.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Saved {_results}/flux1d_map.png, flux1d_J.png, flux1d_Jsym.png')


if __name__ == '__main__':
    save_results()
