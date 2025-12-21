import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


os.environ["CC"]  = "gcc"
os.environ["CXX"] = "g++"


class CustomBuildExt(build_ext):
  def build_extensions(self):
    opts = ["-Ofast", "-std=c++14", "-fPIC"]
    for ext in self.extensions:
      ext.extra_compile_args = opts
    super().build_extensions()


ext_modules = [
  Pybind11Extension(
    "ouxinfo",
    ["src/ouxinfo.cpp"],
    include_dirs=["src"],
    cxx_std=14,
    extra_compile_args=["-Ofast"]
  ),
]

setup(
  name="ouxinfo",
  version="0.1.0",
  description="Fast Shannon entropy estimator using C++",
  ext_modules=ext_modules,
  cmdclass={"build_ext": CustomBuildExt},
  zip_safe=False,
  python_requires=">=3.13",
)

