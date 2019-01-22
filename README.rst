hmmlearn
========

|PyPI| |Read the Docs| |Travis| |AppVeyor| |CodeCov|

.. |PyPI|
   image:: https://img.shields.io/pypi/v/hmmlearn.svg
   :target: https://pypi.python.org/pypi/hmmlearn
.. |Read the Docs|
   image:: https://readthedocs.org/projects/hmmlearn/badge/?version=latest
   :target: http://hmmlearn.readthedocs.io/en/latest/?badge=latest
.. |Travis|
   image:: https://travis-ci.org/hmmlearn/hmmlearn.svg?branch=master
   :target: https://travis-ci.org/hmmlearn/hmmlearn
.. |AppVeyor|
   image:: https://ci.appveyor.com/api/projects/status/github/hmmlearn/hmmlearn?branch=master&svg=true
   :target: https://ci.appveyor.com/project/superbobry/hmmlearn
.. |CodeCov|
   image:: https://codecov.io/gh/hmmlearn/hmmlearn/master.svg
   :target: https://codecov.io/gh/hmmlearn/hmmlearn

hmmlearn is a set of algorithms for **unsupervised** learning and inference
of Hidden Markov Models. For supervised learning learning of HMMs and similar
models see seqlearn_.

.. _seqlearn: https://github.com/larsmans/seqlearn

**Note**: this package has currently no maintainer. Nobody will answer
questions. In particular, the person who is making this code available on
Github will not answer questions, fix bugs, or maintain the package in any way.

If you are interested in contributing, or fixing bugs, please open an issue on
Github and we will gladly give you contributor rights.

Important links
===============

* Official source code repo: https://github.com/hmmlearn/hmmlearn
* HTML documentation (stable release): https://hmmlearn.readthedocs.org/en/stable
* HTML documentation (development version): https://hmmlearn.readthedocs.org/en/latest

Dependencies
============

The required dependencies to use hmmlearn are

* Python >= 2.7
* NumPy >= 1.10
* scikit-learn >= 0.16

You also need Matplotlib >= 1.1.1 to run the examples and pytest >= 2.6.0 to run
the tests.

Installation
============

Requires a C compiler.

::

    pip install --upgrade --user hmmlearn
