#! /usr/bin/env python
#
# Copyright (C) 2007-2009 Cournapeau David <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2014 Gael Varoquaux
#               2014 Sergei Lebedev <superbobry@gmail.com>

"""Hidden Markov Models in Python with scikit-learn like API"""

import sys

from setuptools import setup, Extension
from Cython.Distutils import build_ext


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
    "Programming Language :: Python :: 2.6",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
]

import hmmlearn

VERSION = hmmlearn.__version__


setup_options = dict(
    name="hmmlearn",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    url="https://github.com/hmmlearn/hmmlearn",
    packages=["hmmlearn", "hmmlearn.utils"],
    classifiers=CLASSIFIERS,
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension("hmmlearn._hmmc", ["hmmlearn/_hmmc.pyx"])
    ],
    requires=["sklearn"],
    install_requires=["Cython"]
)


# For these actions, NumPy is not required. We want them to succeed without,
# for example when pip is used to install seqlearn without NumPy present.
NO_NUMPY_ACTIONS = ('--help-commands', 'egg_info', '--version', 'clean')
if not ('--help' in sys.argv[1:]
        or len(sys.argv) > 1 and sys.argv[1] in NO_NUMPY_ACTIONS):
    import numpy as np
    setup_options['include_dirs'] = [np.get_include()]

setup(**setup_options)
