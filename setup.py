import os
import platform
import re
import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


def _get_version():
    text = (this_directory / "ouxinfo" / "_version.py").read_text()
    return re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text).group(1)


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

    if sys.platform == "win32":
      compile_args = ["/O2", "/std:c++14", "/openmp"]
      link_args = []
    elif sys.platform == "darwin":
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
      ext.extra_link_args    = link_args
    super().build_extensions()

ext_modules = [
  Pybind11Extension(
    "ouxinfo._core",
    ["ouxinfo/ouxinfo.cpp"],
    include_dirs=["ouxinfo", "third_party", "third_party/nanoflann"],
    cxx_std=14,
  ),
]


setup(
  name="ouxinfo",
  version=_get_version(),
  packages=["ouxinfo"],
  description="Fast Shannon entropy estimator using C++",
  long_description=long_description,
  long_description_content_type="text/markdown",

  ext_modules=ext_modules,
  cmdclass={"build_ext": CustomBuildExt},
  zip_safe=False,
  python_requires=">=3.10",
  install_requires=[
    'numpy<=2.4',
    'matplotlib',
    'scipy',
    'tqdm',
    'joblib',
  ],
)
