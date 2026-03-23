import os
import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext


os.environ["CC"]  = "gcc"
os.environ["CXX"] = "g++"


# check boost
def check_boost():
  include_candidates = [
    "/usr/include",
    "/usr/local/include",
    os.environ.get("BOOST_ROOT", ""),
  ]
  for inc in include_candidates:
    if inc and os.path.exists(os.path.join(inc, "boost", "version.hpp")):
      return inc
  return None


boost_include = check_boost()
if boost_include is None:
  sys.stderr.write(
    "Boost not found.\n"
    "Please install Boost (e.g. libboost-dev, boost, boost-devel).\n"
  )
  sys.exit(1)


class CustomBuildExt(build_ext):
  def build_extensions(self):
    opts = ["-Ofast", "-march=native", "-mfma", "-fopenmp", "-std=c++14", "-fPIC"]
    for ext in self.extensions:
      ext.extra_compile_args = opts
    super().build_extensions()

ext_modules = [
  Pybind11Extension(
    "ouxinfo._core",
    ["ouxinfo/ouxinfo.cpp"],
    include_dirs=["ouxinfo"],
    cxx_std=14,
    extra_compile_args=["-Ofast", "-march=native", "-mfma", "-fopenmp"],
    extra_link_args=["-fopenmp"],
  ),
]


setup(
  name="ouxinfo",
  version="0.1.0",
  packages=["ouxinfo"],
  description="Fast Shannon entropy estimator using C++",
  ext_modules=ext_modules,
  cmdclass={"build_ext": CustomBuildExt},
  zip_safe=False,
  python_requires=">=3.12",
  install_requires=[
    'pybind11',
    'numpy',
    'matplotlib',
    'scipy',
    'numba',
  ],
)

