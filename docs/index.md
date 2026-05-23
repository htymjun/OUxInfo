[![PyPI version](https://img.shields.io/pypi/v/ouxinfo.svg)](https://pypi.org/project/ouxinfo/)
[![Downloads](https://img.shields.io/pypi/dm/ouxinfo)](https://pypi.org/project/ouxinfo/)
[![DOI](https://zenodo.org/badge/1057221883.svg)](https://doi.org/10.5281/zenodo.19303016)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# OUxInfo

**OUxInfo** is a high-performance Python library for Shannon entropy estimation and information-theoretic causal inference, powered by a C++ backend.

It provides non-parametric k-nearest-neighbor (KSG/Kozachenko-Leonenko) estimators for:

- Shannon entropy, KL divergence, mutual information, conditional mutual information
- Transfer entropy and backward transfer entropy
- Information flow and causal maps
- Directed spatial information flow (1D and 2D)

The C++ core uses [nanoflann](https://github.com/jlblancoc/nanoflann) for fast k-d tree queries and [OpenMP](https://www.openmp.org/) for parallelism, making it suitable for large datasets and multivariate analyses.

---

## Quick Start

```bash
pip install ouxinfo
```

```python
import numpy as np
from ouxinfo import shannon_entropy, transfer_entropy

# Shannon entropy of a standard normal (true value: 0.5 * ln(2πe) ≈ 1.419 nats)
x = np.random.normal(0, 1, 10000).reshape(-1, 1)
H = shannon_entropy(x, k=5)
print(f"H(X) = {H:.3f} nats")

# Transfer entropy from y to x for coupled AR processes
n = 5000
y = np.cumsum(np.random.randn(n)).reshape(-1, 1)
noise = np.random.randn(n).reshape(-1, 1)
x = np.roll(y, 1) * 0.5 + noise * 0.5
TE = transfer_entropy(y, x, tau=1, k=5)
print(f"TE(y -> x) = {TE:.4f} nats/step")
```

---

## Key Features

| Feature | Description |
|---|---|
| **KSG estimators** | k-NN estimators for entropy, MI, CMI, KL divergence |
| **Transfer entropy** | Forward and backward TE with surrogate testing |
| **Information flow** | Horowitz-Esposito rate decomposition (IF, Leak, dI) |
| **Causal maps** | Pairwise TE and IF maps for multivariate systems |
| **Spatial flow** | Directed information flow across 1D/2D spatial grids |
| **TEIFL** | All-in-one analysis: sTE, mTE, bTE, IF, Leak, dI |
| **OpenMP** | Parallelized causal map computation |

---

## Theory Overview

All estimators are based on the **k-nearest-neighbor (KSG)** approach, which is non-parametric and does not assume any distributional form.

**Shannon entropy** is estimated via the Kozachenko-Leonenko formula:

$$H(X) = \psi(N) - \psi(k) + \log c_d + \frac{d}{N}\sum_{i=1}^{N} \log \epsilon(i)$$

**Transfer entropy** quantifies directed information flow from source $Y$ to target $X$:

$$TE_{Y \to X} = I\!\left(X_{t+\tau};\, Y_t^{(m)} \mid X_t^{(m)}\right)$$

**Information flow** decomposes the rate of mutual information change:

$$\frac{d}{dt}I(X_t; Y_t) = \dot{I}_X(t) + \dot{I}_Y(t) + \text{Leak}$$

---


## License

MIT License. Third-party components (nanoflann, Boost) are bundled under their respective BSD/Boost licenses.
