#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

# Newer version is necessary for --only-binary option.
pip install --upgrade pip setuptools

# ``setup.py`` does no fetch dependencies for ``scikit-learn``, install
# them manually. N.B. NumPy >= 1.0 is needed for ``np.broadcast_to`` in
# tests.
pip install --only-binary numpy,scipy 'numpy>=1.10.0' scipy

# XXX currently unused.
if [[ "$COVERAGE" == "true" ]]; then
   pip install pytest-cov coveralls
fi

# Build hmmlearn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
pip install --editable .

python --version
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
