# Plan: Ship Pre-compiled Binary Wheels so Users Don't Need GCC

## Context

`ouxinfo` currently ships only as a source distribution — users must have GCC >= 13 and OpenMP installed before `pip install ouxinfo` can succeed. The fix is to distribute **pre-compiled binary wheels** via PyPI. When pip finds a matching wheel for the user's platform, it downloads the compiled `.so`/`.pyd` directly — no compiler needed. Users on unsupported platforms can still build from the source distribution (sdist), which still requires GCC.

The actual GCC binary is not bundled into the package (it's ~200 MB, platform-specific, and would create legal issues). Instead, CI builds the wheel with GCC so that users receive the compiled output.

---

## Approach: cibuildwheel + GitHub Actions

`cibuildwheel` automates building wheels for Linux (manylinux), macOS (x86_64 + arm64), and Windows. A new workflow fires on version tags, builds all platform wheels, and publishes them to PyPI via OIDC trusted publishing (no secrets needed).

---

## Changes Required

### 1. `setup.py` — Platform-Conditional Compile Flags

The current flags (`-Ofast -mfma -fopenmp -std=c++14 -fPIC`) are Linux/GCC-specific and will break on macOS (no `-fopenmp` in Apple Clang) and Windows (MSVC uses `/openmp`). Also, `-mfma` is x86-only and will break on ARM.

Replace `CustomBuildExt` with:

```python
import sys
import os
import platform

def _target_machine():
    # cibuildwheel sets ARCHFLAGS on macOS for cross-compilation
    archflags = os.environ.get("ARCHFLAGS", "")
    if "arm64" in archflags:
        return "arm64"
    if "x86_64" in archflags:
        return "x86_64"
    return platform.machine().lower()

class CustomBuildExt(build_ext):
    def build_extensions(self):
        machine = _target_machine()
        system = sys.platform

        if system == "win32":
            compile_args = ["/O2", "/openmp", "/std:c++14"]
            link_args = []
        elif system == "darwin":
            compile_args = ["-Ofast", "-fopenmp", "-std=c++14", "-fPIC"]
            link_args = ["-fopenmp"]
            if machine == "x86_64":
                compile_args.insert(1, "-mfma")
        else:  # Linux
            compile_args = ["-Ofast", "-fopenmp", "-std=c++14", "-fPIC"]
            link_args = ["-fopenmp"]
            if machine in ("x86_64", "i686"):
                compile_args.insert(1, "-mfma")

        for ext in self.extensions:
            ext.extra_compile_args = compile_args
            ext.extra_link_args = link_args
        super().build_extensions()
```

Also remove `extra_compile_args` from the `Pybind11Extension(...)` call (line 24) — `CustomBuildExt` overwrites them anyway, and having both is confusing.

### 2. `pyproject.toml` — Add cibuildwheel Config

Append to the existing file:

```toml
[tool.cibuildwheel]
build = "cp312-* cp313-*"
skip = "pp* *-musllinux*"
test-requires = ["pytest", "numpy", "scipy"]
test-command = "pytest {project}/tests/ -v"

[tool.cibuildwheel.linux]
manylinux-x86_64-image = "manylinux_2_28"
manylinux-aarch64-image = "manylinux_2_28"
# manylinux_2_28 = AlmaLinux 8; GCC 13 via gcc-toolset-13 SCL package
before-build = "dnf install -y gcc-toolset-13 gcc-toolset-13-gcc-c++ gcc-toolset-13-libgomp-devel"
environment = { CC = "/opt/rh/gcc-toolset-13/root/usr/bin/gcc", CXX = "/opt/rh/gcc-toolset-13/root/usr/bin/g++" }
# auditwheel repair bundles libgomp.so into the wheel automatically
repair-wheel-command = "auditwheel repair -w {dest_dir} {wheel}"

[tool.cibuildwheel.macos]
archs = ["x86_64", "arm64"]
# Homebrew GCC provides native -fopenmp; Apple Clang does not
before-build = "brew install gcc@13 || true"
environment = { CC = "gcc-13", CXX = "g++-13" }
repair-wheel-command = "delocate-wheel --require-archs {delocate_archs} -w {dest_dir} -v {wheel}"

[tool.cibuildwheel.windows]
archs = ["AMD64"]
# MSVC is the default; setup.py detects win32 and uses /openmp instead of -fopenmp
```

**Why manylinux_2_28**: It's based on AlmaLinux 8 and provides GCC 13 via `gcc-toolset-13`. `auditwheel repair` then bundles `libgomp.so.1` into the wheel, so end-users don't need libgomp installed. The older `manylinux2014` image only has GCC 10.

### 3. `.github/workflows/build_wheels.yml` — New Workflow (create this file)

```yaml
name: Build and Publish Wheels

on:
  push:
    tags:
      - "v*"
  workflow_dispatch:

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-13, macos-14, windows-latest]
        # macos-13 = Intel x86_64, macos-14 = Apple Silicon arm64 (M1 native runner)
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install cibuildwheel
        run: pip install cibuildwheel
      - name: Build wheels
        run: python -m cibuildwheel --output-dir wheelhouse
        env:
          CIBW_ARCHS_MACOS: ${{ matrix.os == 'macos-13' && 'x86_64' || 'arm64' }}
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: ./wheelhouse/*.whl

  build_sdist:
    name: Build sdist
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build && python -m build --sdist
      - uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*.tar.gz

  publish:
    name: Publish to PyPI
    needs: [build_wheels, build_sdist]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          merge-multiple: true
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

**Why two macOS runners instead of cross-compilation**: `brew install gcc@13` installs a native binary for the current arch. An x86_64 runner can't cross-compile arm64 with Homebrew GCC, so using the native M1 runner (`macos-14`) is simpler and more reliable.

---

## One-Time Manual Step: PyPI Trusted Publisher

Before the first tag push, configure OIDC on PyPI:
1. Log in to pypi.org → project `ouxinfo` → Publishing → Add Trusted Publisher
2. GitHub Actions, owner: `htymjun`, repo: `ShannonEntropyEstimator`, workflow: `build_wheels.yml`

This allows publishing with no API tokens or passwords stored as secrets.

---

## Release Process (After Implementation)

```bash
# Bump version in setup.py, then:
git commit -am "Release 0.1.4"
git tag v0.1.4
git push origin v0.1.4
# build_wheels.yml fires → builds all platform wheels → publishes to PyPI
```

---

## Verification

1. Trigger `build_wheels.yml` manually (workflow_dispatch) on a test branch before tagging
2. Download the Linux wheel artifact and confirm: `python -c "import ouxinfo; print(ouxinfo.__version__)"`
3. Confirm `libgomp.so` is bundled: `unzip -l *.whl | grep libgomp`
4. After publishing, test on a machine without GCC: `pip install ouxinfo && python -c "from ouxinfo import H; print(H([1,2,3]))"`

---

## Files Modified / Created

| File | Action |
|---|---|
| `setup.py` | Modify — platform-conditional flags |
| `pyproject.toml` | Modify — add `[tool.cibuildwheel]` sections |
| `.github/workflows/build_wheels.yml` | Create — wheel build + publish CI |
| `.github/workflows/test.yml` | No change needed |
