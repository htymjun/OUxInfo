import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

infomeasure = pytest.importorskip("infomeasure")
from infomeasure import mutual_information as mi_inf
from infomeasure import transfer_entropy as te_inf
from ouxinfo import mutual_info, transfer_entropy

N_VALUES = [1000, 2000, 5000, 10000]
K = 5


def _time_call(fn, *args, **kwargs):
    fn(*args, **kwargs)  # warmup
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def test_speed_comparison():
    rng = np.random.default_rng(0)
    times_mi_oux, times_mi_inf = [], []
    times_te_oux, times_te_inf = [], []

    for N in N_VALUES:
        x = rng.standard_normal(N)
        y = x + rng.standard_normal(N) * 0.5
        times_mi_oux.append(_time_call(mutual_info, x.reshape(-1, 1), y.reshape(-1, 1), k=K))
        times_mi_inf.append(_time_call(mi_inf, x, y, approach="ksg", k=K))

        eps = rng.standard_normal(N)
        y_te = np.roll(x, 1) + eps
        times_te_oux.append(_time_call(transfer_entropy, x.reshape(-1, 1), y_te.reshape(-1, 1),
                                       k=K, tau=1, m=1, lag=1, trial=0))
        times_te_inf.append(_time_call(te_inf, x, y_te, k=K, noise_level=0, prop_time=1, approach='ksg'))

    assert times_mi_oux[-1] < times_mi_inf[-1], (
        f"ouxinfo MI ({times_mi_oux[-1]:.3f}s) not faster than infomeasure "
        f"({times_mi_inf[-1]:.3f}s) at N={N_VALUES[-1]}")

    assert times_te_oux[-1] < times_te_inf[-1], (
        f"ouxinfo TE ({times_te_oux[-1]:.3f}s) not faster than infomeasure "
        f"({times_te_inf[-1]:.3f}s) at N={N_VALUES[-1]}")

    plt.rcParams.update({
        'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'xtick.direction': 'in', 'ytick.direction': 'in', 'font.size': 16,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, t_oux, t_inf, title in [
        (axes[0], times_mi_oux, times_mi_inf, "Mutual Information"),
        (axes[1], times_te_oux, times_te_inf, "Transfer Entropy"),
    ]:
        ax.loglog(N_VALUES, t_oux, 'o-', label='ouxinfo')
        ax.loglog(N_VALUES, t_inf, 's-', label='infomeasure')
        ax.set_title(title)
        ax.set_xlabel(r'$N$')
        ax.set_ylabel('Time (s)')
        ax.legend(frameon=False)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'speed_comparison_omp.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
