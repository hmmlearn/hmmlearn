hmmlearn |travis| |appveyor|
========

.. |travis| image:: https://api.travis-ci.org/hmmlearn/hmmlearn.png?branch=master
   :target: https://travis-ci.org/hmmlearn/hmmlearn

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/3c70msixtdvvae20/branch/master?svg=true
   :target: https://ci.appveyor.com/project/superbobry/hmmlearn/branch/master

hmmlearn is a set of algorithms for **unsupervised** learning and inference of
Hidden Markov Models. For supervised learning learning of HMMs and similar models
see `seqlearn <https://github.com/larsmans/seqlearn>`_.

Important links
===============

* Official source code repo: https://github.com/hmmlearn/hmmlearn
* HTML documentation (stable release): https://hmmlearn.readthedocs.org/en/stable
* HTML documentation (development version): https://hmmlearn.readthedocs.org/en/latest

Dependencies
============

The required dependencies to use hmmlearn are

* Python >= 2.6
* NumPy (tested to work with >=1.9.3)
* SciPy (tested to work with >=0.16.0)
* scikit-learn >= 0.16

You also need Matplotlib >= 1.1.1 to run the examples and pytest >= 2.6.0 to run
the tests.

Installation
============

First make sure you have installed all the dependencies listed above. Then run
the following command::

    pip install -U --user hmmlearn

..
   Development
   ===========

   Detailed instructions on how to contribute are available in
   ``CONTRIBUTING.rst``.
