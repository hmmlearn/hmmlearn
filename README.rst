hmmlearn |Travis|_
========

.. |Travis| image:: https://api.travis-ci.org/hmmlearn/hmmlearn.png?branch=master
.. _Travis: https://travis-ci.org/hmmlearn/hmmlearn

``hmmlearn`` is a set of algorithm for learning and inference of Hiden Markov
Models.

Historically, this code was present in ``scikit-learn``, but unmaintained. It
has been orphaned and separated as a different package.

The learning algorithms in this package are **unsupervised**. For supervised
learning of HMMs and similar models, see `seqlearn
<https://github.com/larsmans/seqlearn>`_.

Getting the latest code
=======================

To get the latest code using git, simply type::

    $ git clone git://github.com/hmmlearn/hmmlearn.git

Installing
==========

Make sure you have all the dependencies::

    $ pip install scikit-learn Cython

and then install ``hmmlearn`` by running::

    $ python setup.py install

in the source code directory.

Running the test suite
======================

To run the test suite, you need ``nosetests`` and the ``coverage`` modules.
Run the test suite using::

    $ python setup.py build_ext --inplace && nosetests

from the root of the project.

Building the docs
=================

To build the docs you need to have the following packages installed::

    $ pip install Pillow matplotlib Sphinx numpydoc

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
