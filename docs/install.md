# Installation

## Requirements
- Python >= 3.10 (>= 3.12 recommended)
- GCC >= 11 (>= 13 recommended)

## Boost C++ Libraries
Boost is now bundled in the `third_party/` directory. You do not need to install Boost separately.

## Install via pip
```bash
pip install ouxinfo
```

## Install via conda-forge
```bash
conda install -c conda-forge ouxinfo
```

## Install via mamba
```bash
mamba install -c conda-forge ouxinfo
```

## Install via pixi
```bash
pixi add ouxinfo
```

## Install locally
```bash
pip install .
```

## Python dependencies
These are installed automatically via pip:
- pybind11
- numpy
- scipy
- matplotlib
- joblib
- tqdm

## C++ dependencies
- Boost (bundled in `third_party/`)
- nanoflann (included in this `ouxinfo/`)

## Notes
- No additional system-level installation of Boost is required.
- The build process will automatically use the bundled Boost headers.
