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
* Boost C++ Libraries (Boost Software License, included in this repository)
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
