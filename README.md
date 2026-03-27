# OUxInfo
OUxInfo is a C++ based fast Shannon entropy estimator for Python.

## Dependencies
* Python (>=3.12 is better)
  * pybind11 (will be installed automatically)
* gcc >= 11 (14 is better)
  * Boost
    ~~~bash
    $ sudo apt install libboost-dev
    ~~~
  * nanoflann (included in this repository)

## Setup
1. Compile OUxInfo
    ~~~bash
    $ pip install .
    ~~~

## License
This project uses nanoflann (BSD license)
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2025  Jose Luis Blanco (joseluisblancoc@gmail.com).
