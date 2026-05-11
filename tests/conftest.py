import matplotlib
matplotlib.use('Agg')  # headless backend — required for CI (no display)


def pytest_sessionfinish(session, exitstatus):
    from test_H      import save_results as _h
    from test_KL     import save_results as _kl
    from test_MI     import save_results as _mi
    from test_IF     import save_results as _if
    from test_TE     import save_results as _te
    from test_flux1d import save_results as _flux1d
    for fn in [_h, _kl, _mi, _if, _te, _flux1d]:
        fn()
