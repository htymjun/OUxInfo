# Installation

This guide explains how to install the OUxInfo Shannon Entropy Estimator package.

## Requirements
- Python >= 3.10 (>= 3.12 recommended)
- GCC >= 11 (>= 13 recommended)
- pip (Python package manager)
- Boost C++ Libraries (not bundled)

### Install Boost on Ubuntu
```bash
sudo apt install libboost-dev
```

## Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ShannonEntropyEstimator.git
   cd ShannonEntropyEstimator
   ```
2. Install the package:
   ```bash
   pip install .
   ```

## Dependencies
- Python: pybind11, numpy, scipy, matplotlib, numba (installed automatically)
- C++: Boost (user must install), nanoflann (included)

For development, install with:
```bash
pip install -e .[dev]
```
