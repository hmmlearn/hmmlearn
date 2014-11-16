HMMLearn: Hidden Markov Models in Python, with scikit-learn like API
=====================================================================


HMMlearn is a set of algorithm for learning and inference of Hiden Markov
Models.

Historically, this code was present in scikit-learn, but unmaintained. It
has been orphaned and separated as a different package.

**Note**: this package has currently no maintainer. Nobody will answer
questions. In particular, the person who is making this code available on
Github will not answer questions, fix bugs, or maintain the package in
any way.

If you are interested in contributing, or fixing bugs, please open an
issue on Github and we will gladly give you contributor rights.

Continuous integration (ie running tests) is found on:
https://travis-ci.org/hmmlearn/hmmlearn

The learning algorithms in this package are unsupervised. For supervised
learning of HMMs and similar models, see `seqlearn
<https://github.com/larsmans/seqlearn>`_.

Getting the latest code
=========================

To get the latest code using git, simply type::

    git clone git://github.com/hmmlearn/hmmlearn.git

Installing
=========================

As any Python packages, to install hmmlearn, simply do::

    python setup.py install

in the source code directory.

HMMLearn depends on scikit-learn.

Running the test suite
=========================

To run the test suite, you need nosetests and the coverage modules.
Run the test suite using::

    python setup.py build_ext --inplace && nosetests

from the root of the project.

Building the docs
=========================

To build the docs you need to have setuptools and sphinx (>=0.5) installed. 
Run the command::

    cd doc
    make html

The docs are built in the build/sphinx/html directory.


Making a source tarball
=========================

To create a source tarball, eg for packaging or distributing, run the
following command::

    python setup.py sdist

The tarball will be created in the `dist` directory. This command will
compile the docs, and the resulting tarball can be installed with
no extra dependencies than the Python standard library. You will need
setuptool and sphinx.

Making a release and uploading it to PyPI
==================================================

This command is only run by project manager, to make a release, and
upload in to PyPI::

    python setup.py sdist bdist_egg register upload


