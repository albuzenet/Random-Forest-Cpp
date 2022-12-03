from glob import glob

import pybind11
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "example1",
        sources=glob("src/*.cpp"),
        include_dirs=[*glob("include/"), pybind11.get_include()],
        define_macros=[("PYBIND11", "1")],
        language="c++",
    ),
]

setup(name="example1", ext_modules=ext_modules)