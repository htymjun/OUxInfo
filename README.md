[![PyPI version](https://img.shields.io/pypi/v/ouxinfo.svg)](https://pypi.org/project/ouxinfo/)
[![Downloads](https://img.shields.io/pypi/dm/ouxinfo)](https://pypi.org/project/ouxinfo/)
[![DOI](https://zenodo.org/badge/1057221883.svg)](https://doi.org/10.5281/zenodo.19303016)

# OUxInfo
OUxInfo is a high-performance Shannon entropy estimator for Python, powered by a C++ backend.
It is designed for fast and scalable entropy estimation, particularly for causal inference.

## Features
* Fast Shannon entropy estimator (C++ backend)
* Python interface via pybind11
* Information-theoretic quantities for causal inference (e.g., transfer entropy, backward transfer entropy, information flow)

## Installation
### Requirements
* Python >= 3.12 (3.13 recommended)
* GCC >= 11 (>= 13 recommended)
* OpenMP support (-fopenmp)

### Install via pip
* PyPI
~~~bash
$ pip install ouxinfo
~~~
* Clone this repository and
~~~bash
$ pip install .
~~~

### Dependencies
#### Python packages (installed automatically via pip)
* pybind11
* numpy
* scipy
* matplotlib
* numba

#### C++ dependencies
* Boost C++ Libraries (Boost Software License, included in this repository)
* nanoflann (BSD License, included in this repository)

## Usage (example)
~~~bash
import numpy as np
from ouxinfo import shannon_entropy

x = np.random.normal(0.e0, 1.e0, 10000)
H = shannon_entropy(x.reshape(-1,1), k=5)
~~~

## Related Publication
This repository contains the implementation used in the following publication:

Jun Hatayama, Kento Tanaka, and Toshinori Kouchi. "Nonlinear causal relationship between separation bubbles and reflected shock wave in shock wave/turbulent boundary layer interaction based on information theory." Computers & Fluids (2026): 107016.

~~~bash
@article{hatayama2026nonlinear,
  title={Nonlinear causal relationship between separation bubbles and reflected shock wave in shock wave/turbulent boundary layer interaction based on information theory},
  author={Hatayama, Jun and Tanaka, Kento and Kouchi, Toshinori},
  journal={Computers \& Fluids},
  pages={107016},
  year={2026},
  publisher={Elsevier}
}
~~~

The repository was made publicly available after publication to improve reproducibility.
However, this version may differ slightly from the version used in the paper.

## License
This project is released under MIT license.

## Third-party licenses
This project depends on the following libraries;
### nanoflann (BSD License)
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2025  Jose Luis Blanco (joseluisblancoc@gmail.com).
Redistribution and use in source and binary forms are permitted under the BSD License.

### Boost C++ Libraries (Boost Software License 1.0)
 * Copyright John Maddock 2006, 2007.
 * Copyright Paul A. Bristow 2006, 2007, 2009, 2010.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Boost is now bundled in the `third_party/` directory and is used automatically during build.

### pybind11 (BSD License)
Used for Python bindings.

### NumPy, SciPy, Matplotlib, Numba
These libraries are installed via pip and are subject to their respective licenses.
