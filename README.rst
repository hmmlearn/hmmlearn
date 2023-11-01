hmmlearn
========

| |GitHub| |PyPI|
| |Read the Docs| |Build| |CodeCov|

.. |GitHub|
   image:: https://img.shields.io/badge/github-hmmlearn%2Fhmmlearn-brightgreen
   :target: https://github.com/hmmlearn/hmmlearn
.. |PyPI|
   image:: https://img.shields.io/pypi/v/hmmlearn.svg?color=brightgreen
   :target: https://pypi.python.org/pypi/hmmlearn
.. |Read the Docs|
   image:: https://readthedocs.org/projects/hmmlearn/badge/?version=latest
   :target: http://hmmlearn.readthedocs.io/en/latest/?badge=latest
.. |Build|
   image:: https://img.shields.io/github/actions/workflow/status/hmmlearn/hmmlearn/build.yml?branch=main
   :target: https://github.com/hmmlearn/hmmlearn/actions
.. |CodeCov|
   image:: https://img.shields.io/codecov/c/github/hmmlearn/hmmlearn
   :target: https://codecov.io/gh/hmmlearn/hmmlearn

hmmlearn is a set of algorithms for **unsupervised** learning and inference
of Hidden Markov Models. For supervised learning learning of HMMs and similar
models see seqlearn_.

.. _seqlearn: https://github.com/larsmans/seqlearn

**Note**: This package is under limited-maintenance mode.

Important links
===============

* Official source code repo: https://github.com/hmmlearn/hmmlearn
* HTML documentation (stable release): https://hmmlearn.readthedocs.org/en/stable
* HTML documentation (development version): https://hmmlearn.readthedocs.org/en/latest

Dependencies
============

The required dependencies to use hmmlearn are

* Python >= 3.6
* NumPy >= 1.10
* scikit-learn >= 0.16

You also need Matplotlib >= 1.1.1 to run the examples and pytest >= 2.6.0 to run
the tests.

Installation
============

Requires a C compiler and Python headers.

To install from PyPI::

    pip install --upgrade --user hmmlearn

To install from the repo::

    pip install --user git+https://github.com/hmmlearn/hmmlearn
