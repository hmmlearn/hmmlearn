hmmlearn
========

|GitHub| |PyPI|

|Read the Docs| |Azure Pipelines| |CodeCov|

.. |GitHub|
   image:: https://img.shields.io/badge/github-hmmlearn%2Fhmmlearn-brightgreen
   :target: https://github.com/hmmlearn/hmmlearn
.. |PyPI|
   image:: https://img.shields.io/pypi/v/hmmlearn.svg
   :target: https://pypi.python.org/pypi/hmmlearn
.. |Read the Docs|
   image:: https://readthedocs.org/projects/hmmlearn/badge/?version=latest
   :target: http://hmmlearn.readthedocs.io/en/latest/?badge=latest
.. |Azure Pipelines|
   image:: https://dev.azure.com/anntzer/hmmlearn/_apis/build/status/anntzer.hmmlearn
   :target: https://dev.azure.com/anntzer/hmmlearn/_build/latest?definitionId=1
.. |CodeCov|
   image:: https://codecov.io/gh/hmmlearn/hmmlearn/master.svg
   :target: https://codecov.io/gh/hmmlearn/hmmlearn

hmmlearn is a set of algorithms for **unsupervised** learning and inference
of Hidden Markov Models. For supervised learning learning of HMMs and similar
models see seqlearn_.

.. _seqlearn: https://github.com/larsmans/seqlearn

**Note**: This package is under limited-maintenance mode.
Moreover, if you are able to help with testing on macOS, please have a look at
https://github.com/hmmlearn/hmmlearn/issues/370; your help will be greatly
appreciated.

Important links
===============

* Official source code repo: https://github.com/hmmlearn/hmmlearn
* HTML documentation (stable release): https://hmmlearn.readthedocs.org/en/stable
* HTML documentation (development version): https://hmmlearn.readthedocs.org/en/latest

Dependencies
============

The required dependencies to use hmmlearn are

* Python >= 3.5
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
