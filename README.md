# OUxInfo
OUxInfo is a high-performance Shannon entropy estimator for Python, powered by a C++ backend.
It is designed for fast and scalable entropy estimation, particularly for causal inference.

## Features
* Fast Shannon entropy estimator (C++ backend)
* Python interface via pybind11
* Information-theoretic quantities for causal inference (e.g., transfer entropy, backward transfer entropy, information flow)

## Installation
### Requirements
* Python >= 3.10 (>= 3.12 recommended)
* GCC >= 11 (>= 13 recommended)
* Boost C++ Libraries (not bundled)
#### Install Boost on Ubuntu:
~~~bash
$ sudo apt install libboost-dev
~~~

### Install via pip
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
* Boost C++ Libraries (Boost Software License)
* nanoflann (BSD License, included in this repository)

## Usage (example)
~~~bash
import numpy as np
from ouxinfo import shannon_entropy

x = np.random.normal(0.e0, 1.e0, 10000)
H = shannon_entropy(x.reshape(-1,1), k=5)
~~~

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
Boost is not distributed with this package and must be installed separately by the user.

### pybind11 (BSD License)
Used for Python bindings.

### NumPy, SciPy, Matplotlib, Numba
These libraries are installed via pip and are subject to their respective licenses.
