from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


class CustomBuildExt(build_ext):
  def build_extensions(self):
    opts = ["-Ofast", "-mfma", "-fopenmp", "-std=c++14", "-fPIC"]
    for ext in self.extensions:
      ext.extra_compile_args = opts
      ext.extra_link_args    = ["-fopenmp"]
    super().build_extensions()

ext_modules = [
  Pybind11Extension(
    "ouxinfo._core",
    ["ouxinfo/ouxinfo.cpp"],
    include_dirs=["ouxinfo", "third_party"],
    cxx_std=14,
    extra_compile_args=["-Ofast", "-mfma", "-fopenmp"],
  ),
]


setup(
  name="ouxinfo",
  version="0.1.1",
  packages=["ouxinfo"],
  description="Fast Shannon entropy estimator using C++",
  long_description=long_description,
  long_description_content_type="text/markdown",

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

