hmmlearn |travis| |appveyor|
========

.. |travis| image:: https://api.travis-ci.org/hmmlearn/hmmlearn.png?branch=master
   :target: https://travis-ci.org/hmmlearn/hmmlearn

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/3c70msixtdvvae20/branch/master?svg=true
   :target: https://ci.appveyor.com/project/superbobry/hmmlearn/branch/master

hmmlearn is a set of algorithms for **unsupervised** learning and inference of
Hidden Markov Models. For supervised learning learning of HMMs and similar models
see `seqlearn <https://github.com/larsmans/seqlearn>`_.

Getting the latest code
=======================

To get the latest code using git, simply type::

    $ git clone https://github.com/hmmlearn/hmmlearn.git

Installing
==========

Make sure you have all the dependencies::

    $ pip install numpy scipy scikit-learn

and then install ``hmmlearn`` by running::

    $ python setup.py install

in the source code directory.

Running the test suite
======================

To run the test suite, you need ``pytest``. Run the test suite using::

    $ python setup.py build_ext --inplace
    $ py.test --doctest-modules hmmlearn

from the root of the project.

Building the docs
=================

To build the docs you need to have the following packages installed::

    $ pip install Pillow matplotlib Sphinx sphinx-gallery sphinx_rtd_theme numpydoc

Run the command::

    $ cd doc
    $ make html

The docs are built in the ``_build/html`` directory.

Making a source tarball
=======================

To create a source tarball, eg for packaging or distributing, run the
following command::

    $ python setup.py sdist

The tarball will be created in the ``dist`` directory.

Making a release and uploading it to PyPI
=========================================

This command is only run by project manager, to make a release, and
upload in to PyPI::

    $ python setup.py sdist bdist_egg register upload
