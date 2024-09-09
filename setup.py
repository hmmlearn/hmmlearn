from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup


setup(
    ext_modules=[
        Pybind11Extension("hmmlearn._hmmc", ["ext/_hmmc.cpp"], cxx_std=11),
    ],
)
