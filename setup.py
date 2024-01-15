# Copyright (C) 2007-2009 Cournapeau David <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2014 Gael Varoquaux
#               2014-2016 Sergei Lebedev <superbobry@gmail.com>
#               2018- Antony Lee

import os
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class build_ext(build_ext):

    def finalize_options(self):
        from pybind11.setup_helpers import Pybind11Extension
        self.distribution.ext_modules[:] = [Pybind11Extension(
            "hmmlearn._hmmc", ["src/_hmmc.cpp"], cxx_std=11)]
        super().finalize_options()

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        super().build_extensions()


setup(
    name="hmmlearn",
    description="Hidden Markov Models in Python with scikit-learn like API",
    long_description=open("README.rst", encoding="utf-8").read(),
    long_description_content_type="text/x-rst",
    maintainer="Antony Lee",
    url="https://github.com/hmmlearn/hmmlearn",
    license="new BSD",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={"build_ext": build_ext},
    py_modules=[],
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    ext_modules=[Extension("", [])],
    package_data={},
    python_requires=">=3.8",
    setup_requires=[
        "pybind11>=2.6",
        "setuptools_scm>=3.3",  # fallback_version.
    ],
    use_scm_version=lambda: {  # xref __init__.py
        "version_scheme": "post-release",
        "local_scheme": "node-and-date",
        "write_to": "lib/hmmlearn/_version.py",
        "fallback_version": "0+unknown",
    },
    install_requires=[
        "numpy>=1.10",  # np.broadcast_to.
        "scikit-learn>=0.16,!=0.22.0",  # check_array, check_is_fitted.
        "scipy>=0.19",  # scipy.special.logsumexp.
    ],
    extras_require={
        "tests": ["pytest"],
        "docs": [
            "matplotlib",
            "pydata_sphinx_theme",
            "sphinx>=2.0",
            "sphinx-gallery",
        ],
    },
    entry_points={
        "console_scripts": [],
        "gui_scripts": [],
    },
)
