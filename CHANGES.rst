hmmlearn Changelog
==================

Here you can see the full list of changes between each hmmlearn release.

Version 0.2.0
-------------

Relased on March 1st, 2016

The release contains a known bug: fitting ``GMMHMM`` with covariance
types other than ``"diag"`` does not work. This is going to be fixed
in the following version. See issue #78 on GitHub for details.

- Removed deprecated re-exports from ``hmmlean.hmm``.
- Speed up forward-backward algorithms and Viterbi decoding by using Cython
  typed memoryviews. Thanks to @cfarrow. See PR#82 on GitHub.
- Changed the API to accept multiple sequences via a single feature matrix
  ``X`` and an array of sequence ``lengths``. This allowed to use the HMMs
  as part of scikit-learn ``Pipeline``. The idea was shamelessly plugged
  from ``seqlearn`` package by @larsmans. See issue #29 on GitHub.
- Removed ``params`` and ``init_params`` from internal methods. Accepting
  these as arguments was redundant and confusing, because both available
  as instance attributes.
- Implemented ``ConvergenceMonitor``, a class for convergence diagnostics.
  The idea is due to @mvictor212.
- Added support for non-fully connected architectures, e.g. left-right HMMs.
  Thanks to @matthiasplappert. See issue #33 and PR #38 on GitHub.
- Fixed normalization of emission probabilities in ``MultinomialHMM``, see
  issue #19 on GitHub.
- ``GaussianHMM`` is now initialized from all observations, see issue #1 on GitHub.
- Changed the models to do input validation lazily as suggested by the
  scikit-learn guidelines.
- Added ``min_covar`` parameter for controlling overfitting of ``GaussianHMM``,
  see issue #2 on GitHub.
- Accelerated M-step fro `GaussianHMM` with full and tied covariances. See
  PR #97 on GitHub. Thanks to @anntzer.
- Fixed M-step for ``GMMHMM``, which incorrectly expeced ``GMM.score_samples``
  to return log-probabilities. See PR #4 on GitHub for discussion. Thanks to
  @mvictor212 and @michcio1234.

Version 0.1.1
-------------

Initial release, released on February 9th 2015
