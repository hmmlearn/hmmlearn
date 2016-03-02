#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip pytest \
    scikit-learn cython
source activate testenv

if [[ "$COVERAGE" == "true" ]]; then
    pip install pytest-cov coveralls
fi

# Build hmmlearn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python setup.py build_ext --inplace
