#! /usr/bin/env python
#
# Copyright (C) 2007-2009 Cournapeau David <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2014 Gael Varoquaux
#               2014-2016 Sergei Lebedev <superbobry@gmail.com>

"""Hidden Markov Models in Python with scikit-learn like API"""

import sys

try:
    from numpy.distutils.misc_util import get_info
except ImportError:
    # A dirty hack to get RTD running.
    def get_info(name):
        return {}

from setuptools import setup, Extension


DISTNAME = "hmmlearn"
DESCRIPTION = __doc__
LONG_DESCRIPTION = open("README.rst").read()
MAINTAINER = "Sergei Lebedev"
MAINTAINER_EMAIL = "superbobry@gmail.com"
LICENSE = "new BSD"

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Cython",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
]

import hmmlearn

VERSION = hmmlearn.__version__

install_requires = ["numpy", "scikit-learn>=0.16"]
tests_require = install_requires + ["pytest"]
docs_require = install_requires + [
    "Sphinx", "sphinx-gallery", "numpydoc", "Pillow", "matplotlib"
]

setup_options = dict(
    name="hmmlearn",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    url="https://github.com/hmmlearn/hmmlearn",
    packages=["hmmlearn", "hmmlearn.tests"],
    classifiers=CLASSIFIERS,
    ext_modules=[
        Extension("hmmlearn._hmmc", ["hmmlearn/_hmmc.c"],
                  extra_compile_args=["-O3"],
                  **get_info("npymath"))
    ],
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        "tests": tests_require,
        "docs": docs_require
    }
)


if __name__ == "__main__":
    setup(**setup_options)
