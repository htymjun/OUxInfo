# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OUxInfo** — a high-performance Shannon entropy estimator for Python with a C++ backend. Designed for information-theoretic causal inference in dynamical systems (transfer entropy, mutual information, information flow, KL divergence).

## Commands

### Build

```bash
# Full clean rebuild (recommended after C++ changes)
./rebuild.sh

# Manual build
python -m build
pip install dist/*.whl
```

> Requires GCC >= 13 and OpenMP. The C++ backend is compiled with `-Ofast -mfma -fopenmp -std=c++14`.

### Test

```bash
pytest tests/
# Run a single test function
pytest tests/test_H.py::test_name
```

### Install for development

```bash
pip install -e .
```

## Architecture

The project is a hybrid Python/C++ library with three layers:

### 1. C++ core (`ouxinfo/ouxinfo.cpp` + headers)

All estimators use k-nearest-neighbor (KSG/Kozachenko-Leonenko) methods with a Chebyshev-metric k-d tree via `nanoflann.hpp`. OpenMP parallelizes the per-sample loop. Each algorithm is in its own header:

| Header | Algorithm |
|---|---|
| `shannon_entropy.hpp` | Kozachenko-Leonenko entropy |
| `mutual_information.hpp` | Kraskov MI (Type 1) and CMI |
| `kullback_leibler_divergence.hpp` | Pérez-Cruz KL divergence |

`ouxinfo.cpp` also implements transfer entropy, information flow, backward transfer entropy, and multivariate causal maps, then exposes everything to Python via **pybind11** as the `_core` module.

Bundled third-party C++ code lives in `third_party/` (Boost digamma) and `ouxinfo/nanoflann.hpp`.

### 2. Python API (`ouxinfo/`)

- `__init__.py` — public API surface; re-exports from `_core` and the Python modules
- `teifl.py` — high-level `TEIFL()` and `plot_TEIFL()` for complete causal analysis (single-step TE, multi-step TE, backward TE, information flow, leak, sensor capacity)
- `backwardTE.py` — backward transfer entropy (time-reversal analysis)
- `utils.py` — matplotlib configuration helpers

### 3. Scientific models & experiments (not part of the installable package)

- `shell_model/` — GOY and Sabra shell-model turbulence simulations (RK4 integration)
- `experiments/` — benchmarks against infomeasure and PyIF
- `tutorials/` — usage examples

> Ignore `./legacy/` — it is not maintained.

## Key Dependencies

- **pybind11** — Python/C++ bindings
- **numpy, scipy, matplotlib, numba** — installed automatically via `pyproject.toml`
- **Boost** (bundled in `third_party/`), **nanoflann** (bundled in `ouxinfo/`)
- Python >= 3.12, GCC >= 13

## Tests

`tests/test_H.py` validates entropy estimation against theoretical Gaussian entropy and checks robustness across values of `k` (the KNN parameter). Assertions use a 5% tolerance of the true value.

## Planning
- Plans must be saved in `./docs/plans/`
- File names must be in the format `YYYY-MM-DD-<topic>.md`
