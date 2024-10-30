hmmlearn Changelog
==================

Here you can see the full list of changes between each hmmlearn release.

Version 0.3.3
~~~~~~~~~~~~~

Released on October 31, 2024.

- Provide wheels compatible with numpy 2.

Version 0.3.2
-------------

Released on March 1st, 2024.

- update CI/CD Pipelines that were troublesome

Version 0.3.1
-------------

Released on March 1st, 2024.

- Support Python 3.8-3.12
- Improve stability of test suite.  Ensure the documentation examples are covered.
- Documentation Improvements throughout.

Version 0.3.0
-------------

Released on April 18th, 2023.

- Introduce learning HMMs with Variational Inference.  Support
  Gaussian and Categorical Emissions.  This feature is provisional and subject
  to further changes.
- Deprecated support for inputs of shape other than ``(n_samples, 1)`` for
  categorical HMMs.
- Removed the deprecated ``iter_from_X_lengths`` and ``log_mask_zero``;
  ``lengths`` arrays that do not sum up to the entire array length are no
  longer supported.
- Support variable ``n_trials`` in ``MultinomialHMM``, except for sampling.

Version 0.2.8
-------------

Released on September 26th, 2022.

- The ``PoissonHMM`` class was added with an example use case.
- For ``MultinomialHMM``, parameters after ``transmat_prior`` are now
  keyword-only.
- ``startmat_`` and ``transmat_`` will both be initialized with random
  variables drawn from a Dirichlet distribution, to maintain the old
  behavior, these must be initialized as ``1 / n_components``.
- The old ``MultinomialHMM`` class was renamed to ``CategoricalHMM`` (as that's
  what it actually implements), and a new ``MultinomialHMM`` class was
  introduced (with a warning) that actually implements a multinomial
  distribution.
- ``hmmlearn.utils.log_mask_zero`` has been deprecated.

Version 0.2.7
-------------

Released on February 10th, 2022.

- Dropped support for Py3.5 (due to the absence of manylinux wheel supporting
  both Py3.5 and Py3.10).
- ``_BaseHMM`` has been promoted to public API and has been renamed to
  ``BaseHMM``.
- MultinomialHMM no longer overwrites preset ``n_features``.
- An implementation of the Forward-Backward algorithm based upon scaling
  is available by specifying ``implementation="scaling"`` when instantiating
  HMMs. In general, the scaling algorithm is more efficient than an
  implementation based upon logarithms. See `scripts/benchmark.py` for
  a comparison of the performance of the two implementations.
- The *logprob* parameter to `.ConvergenceMonitor.report` has been renamed to
  *log_prob*.

Version 0.2.6
-------------

Released on July 18th, 2021.

- Fixed support for multi-sequence GMM-HMM fit.
- Deprecated ``utils.iter_from_X_lengths``.
- Previously, APIs taking a *lengths* parameter would silently drop the last
  samples if the total length was less than the number of samples.  This
  behavior is deprecated and will raise an exception in the future.

Version 0.2.5
-------------

Released on February 3rd, 2021.

- Fixed typo in implementation of covariance maximization for GMMHMM.
- Changed history of ConvergenceMonitor to include the whole history for
  evaluation purposes.  It can no longer be assumed that it has a maximum
  length of two.

Version 0.2.4
-------------

Released on September 12th, 2020.

.. warning::
   GMMHMM covariance maximization was incorrect in this release.  This bug was
   fixed in the following release.

- Bumped previously incorrect dependency bound on scipy to 0.19.
- Bug fix for 'params' argument usage in GMMHMM.
- Warn when an explicitly set attribute would be overridden by
  ``init_params_``.

Version 0.2.3
-------------

Released on December 17th, 2019.

Fitting of degenerate GMMHMMs appears to fail in certain cases on macOS; help
with troubleshooting would be welcome.

- Dropped support for Py2.7, Py3.4.
- Log warning if not enough data is passed to fit() for a meaningful fit.
- Better handle degenerate fits.
- Allow missing observations in input multinomial data.
- Avoid repeatedly rechecking validity of Gaussian covariance matrices.

Version 0.2.2
-------------

Released on May 5th, 2019.

This version was cut in particular in order to clear up the confusion between
the "real" v0.2.1 and the pseudo-0.2.1 that were previously released by various
third-party packagers.

- Custom ConvergenceMonitors subclasses can be used (#218).
- MultinomialHMM now accepts unsigned symbols (#258).
- The ``get_stationary_distribution`` returns the stationary distribution of
  the transition matrix (i.e., the rescaled left-eigenvector of the transition
  matrix that is associated with the eigenvalue 1) (#141).

Version 0.2.1
-------------

Released on October 17th, 2018.

- GMMHMM was fully rewritten (#107).
- Fixed underflow when dealing with logs. Thanks to @aubreyli. See
  PR #105 on GitHub.
- Reduced worst-case memory consumption of the M-step from O(S^2 T)
  to O(S T). See issue #313 on GitHub.
- Dropped support for Python 2.6. It is no longer supported by
  scikit-learn.

Version 0.2.0
-------------

Released on March 1st, 2016.

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
- Fixed M-step for ``GMMHMM``, which incorrectly expected ``GMM.score_samples``
  to return log-probabilities. See PR #4 on GitHub for discussion. Thanks to
  @mvictor212 and @michcio1234.

Version 0.1.1
-------------

Initial release, released on February 9th 2015.
