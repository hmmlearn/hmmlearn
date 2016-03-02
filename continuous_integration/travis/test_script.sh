#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

# Get into a temp directory to run test from the installed hmmlearn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR
cd $TEST_DIR

python --version
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"

if [[ "$COVERAGE" == "true" ]]; then
    py.test -s -v --cov=hmmlearn --doctest-modules --durations=10 \
            --pyargs hmmlearn
else
    py.test -s -v --doctest-modules --durations=10 --pyargs hmmlearn
fi
