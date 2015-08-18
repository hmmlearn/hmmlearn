# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
# and Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import numpy as np
from sklearn import cluster
from sklearn.mixture import (
    GMM, sample_gaussian,
    log_multivariate_normal_density,
    distribute_covar_matrix_to_match_covariance_type, _validate_covars)
from sklearn.utils import check_random_state

from .base import _BaseHMM
from .utils import iter_from_X_lengths, normalize

__all__ = ['GMMHMM',
           'GaussianHMM',
           'MultinomialHMM']

COVARIANCE_TYPES = ['spherical', 'tied', 'diag', 'full']


class GaussianHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string, one of the :data:`base.DECODER_ALGORITHMS`
        Decoder algorithm.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars_ : array
        Covariance parameters for each state.  The shape depends on
        ``covariance_type``::

            (n_components, )                        if 'spherical',
            (n_features, n_features)                if 'tied',
            (n_components, n_features)              if 'diag',
            (n_components, n_features, n_features)  if 'full'

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianHMM(algorithm='viterbi',...

    See Also
    --------
    GMM : Gaussian mixture model
    """
    def __init__(self, n_components=1, covariance_type='diag',
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.covariance_type = covariance_type
        self.means_prior = means_prior
        self.means_weight = means_weight

        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self.covariance_type == 'full':
            return self._covars_
        elif self.covariance_type == 'diag':
            return np.array([np.diag(cov) for cov in self._covars_])
        elif self.covariance_type == 'tied':
            return np.array([self._covars_] * self.n_components)
        elif self.covariance_type == 'spherical':
            return np.array(
                [np.eye(self.n_features) * cov for cov in self._covars_])

    def _set_covars(self, covars):
        self._covars_ = np.asarray(covars).copy()

    covars_ = property(_get_covars, _set_covars)

    def _check(self):
        super(GaussianHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {0}'
                             .format(COVARIANCE_TYPES))

        _validate_covars(self._covars_, self.covariance_type,
                         self.n_components)

    def _compute_log_likelihood(self, obs):
        return log_multivariate_normal_density(
            obs, self.means_, self._covars_, self.covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        if self.covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self.means_[state], cv, self.covariance_type,
                               random_state=random_state)

    def _init(self, X, lengths=None, params='stmc'):
        super(GaussianHMM, self)._init(X, lengths=lengths, params=params)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if 'c' in params or not hasattr(self, "covars_"):
            cv = np.cov(X.T)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components)
            self._covars_ = self._covars_.copy()
            if self._covars_.any() == 0:
                self._covars_[self._covars_ == 0] = 1e-5

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        if 'm' in params or 'c' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                for t, o in enumerate(obs):
                    obsobsT = np.outer(o, o)
                    for c in range(self.n_components):
                        stats['obs*obs.T'][c] += posteriors[t, c] * obsobsT

    def _do_mstep(self, stats, params):
        super(GaussianHMM, self)._do_mstep(stats, params)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * (meandiff) ** 2
                          + stats['obs**2']
                          - 2 * self.means_ * stats['obs']
                          + self.means_ ** 2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self.means_[c], self.means_[c])
                                 * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self.covariance_type == 'full':
                    self._covars_ = ((covars_prior + cv_num) /
                                     (cvweight + stats['post'][:, None, None]))


class MultinomialHMM(_BaseHMM):
    """Hidden Markov Model with multinomial (discrete) emissions

    Parameters
    ----------

    n_components : int
        Number of states.

    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string, one of the :data:`base.DECODER_ALGORITHMS`
        Decoder algorithm.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the observations).

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    emissionprob_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.hmm import MultinomialHMM
    >>> MultinomialHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    MultinomialHMM(algorithm='viterbi',...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

    def _compute_log_likelihood(self, X):
        return np.log(self.emissionprob_)[:, np.concatenate(X)].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def _init(self, X, lengths=None, params='ste'):
        if not self._check_input_symbols(X):
            raise ValueError("expected a sample from "
                             "a Multinomial distribution.")

        super(MultinomialHMM, self)._init(X, lengths=lengths, params=params)
        self.random_state = check_random_state(self.random_state)

        if 'e' in params:
            if not hasattr(self, "n_features"):
                symbols = set()
                for i, j in iter_from_X_lengths(X, lengths):
                    symbols |= set(X[i:j].flatten())
                self.n_features = len(symbols)
            self.emissionprob_ = self.random_state \
                .rand(self.n_components, self.n_features)
            normalize(self.emissionprob_, axis=1)

    def _check(self):
        super(MultinomialHMM, self)._check()

        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        n_features = getattr(self, "n_features", self.emissionprob_.shape[1])
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features)")
        else:
            self.n_features = n_features

    def _initialize_sufficient_statistics(self):
        stats = super(MultinomialHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MultinomialHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        if 'e' in params:
            for t, symbol in enumerate(np.concatenate(X)):
                stats['obs'][:, symbol] += posteriors[t]

    def _do_mstep(self, stats, params):
        super(MultinomialHMM, self)._do_mstep(stats, params)
        if 'e' in params:
            self.emissionprob_ = (stats['obs']
                                  / stats['obs'].sum(1)[:, np.newaxis])

    def _check_input_symbols(self, X):
        """Check if ``X`` is a sample from a Multinomial distribution.

        That is ``X`` should be an array of non-negative integers from
        range ``[min(X), max(X)]``, such that each integer from the range
        occurs in ``X`` at least once.

        For example ``[0, 0, 2, 1, 3, 1, 1]`` is a valid sample from a
        Multinomial distribution, while ``[0, 0, 3, 5, 10]`` is not.
        """
        symbols = np.concatenate(X)
        if (len(symbols) == 1 or          # not enough data
            symbols.dtype.kind != 'i' or  # not an integer
            (symbols < 0).any()):         # contains negative integers
            return False

        symbols.sort()
        return np.all(np.diff(symbols) <= 1)


class GMMHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian mixture emissions.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    n_mix : int
        Number of states in the GMM.

    covariance_type : string
        String describing the type of covariance parameters for
        the GMM to use.  Must be one of 'spherical', 'tied', 'diag',
        'full'. Defaults to 'diag'.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string, one of the :data:`base.DECODER_ALGORITHMS`
        Decoder algorithm.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    init_params : string, optional
        Controls which parameters are initialized prior to training. Can
        contain any combination of 's' for startprob, 't' for transmat, 'm'
        for means, 'c' for covars, and 'w' for GMM mixing weights.
        Defaults to all parameters.

    params : string, optional
        Controls which parameters are updated in the training process.  Can
        contain any combination of 's' for startprob, 't' for transmat, 'm' for
        means, and 'c' for covars, and 'w' for GMM mixing weights.
        Defaults to all parameters.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    gmms_ : list of GMM objects, length n_components
        GMM emission distributions for each state.

    Examples
    --------
    >>> from hmmlearn.hmm import GMMHMM
    >>> GMMHMM(n_components=2, n_mix=10, covariance_type='diag')
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    GMMHMM(algorithm='viterbi', covariance_type='diag',...

    See Also
    --------
    GaussianHMM : HMM with Gaussian emissions
    """

    def __init__(self, n_components=1, n_mix=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 covariance_type='diag', covars_prior=1e-2,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmcw", init_params="stmcw"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm, random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

        # XXX: Hotfit for n_mix that is incompatible with the scikit's
        # BaseEstimator API
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.covars_prior = covars_prior
        self.gmms_ = []
        for x in range(self.n_components):
            if covariance_type is None:
                gmm = GMM(n_mix)
            else:
                gmm = GMM(n_mix, covariance_type=covariance_type)
            self.gmms_.append(gmm)

    def _compute_log_likelihood(self, X):
        return np.array([g.score(X) for g in self.gmms_]).T

    def _generate_sample_from_state(self, state, random_state=None):
        return self.gmms_[state].sample(1, random_state=random_state).flatten()

    def _init(self, X, lengths=None, params='stwmc'):
        super(GMMHMM, self)._init(X, lengths=lengths, params=params)

        for g in self.gmms_:
            g.set_params(init_params=params, n_iter=0)
            g.fit(X)

    def _initialize_sufficient_statistics(self):
        stats = super(GMMHMM, self)._initialize_sufficient_statistics()
        stats['norm'] = [np.zeros(g.weights_.shape) for g in self.gmms_]
        stats['means'] = [np.zeros(np.shape(g.means_)) for g in self.gmms_]
        stats['covars'] = [np.zeros(np.shape(g.covars_)) for g in self.gmms_]
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(GMMHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        for state, g in enumerate(self.gmms_):
            _, lgmm_posteriors = g.score_samples(X)
            lgmm_posteriors += np.log(posteriors[:, state][:, np.newaxis]
                                      + np.finfo(np.float).eps)
            gmm_posteriors = np.exp(lgmm_posteriors)
            tmp_gmm = GMM(g.n_components, covariance_type=g.covariance_type)
            n_features = g.means_.shape[1]
            tmp_gmm._set_covars(
                distribute_covar_matrix_to_match_covariance_type(
                    np.eye(n_features), g.covariance_type,
                    g.n_components))
            norm = tmp_gmm._do_mstep(X, gmm_posteriors, params)

            if np.any(np.isnan(tmp_gmm.covars_)):
                raise ValueError

            stats['norm'][state] += norm
            if 'm' in params:
                stats['means'][state] += tmp_gmm.means_ * norm[:, np.newaxis]
            if 'c' in params:
                if tmp_gmm.covariance_type == 'tied':
                    stats['covars'][state] += tmp_gmm.covars_ * norm.sum()
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(tmp_gmm.covars_.ndim)
                    shape[0] = np.shape(tmp_gmm.covars_)[0]
                    cvnorm.shape = shape
                    stats['covars'][state] += tmp_gmm.covars_ * cvnorm

    def _do_mstep(self, stats, params):
        super(GMMHMM, self)._do_mstep(stats, params)

        # All that is left to do is to apply covars_prior to the
        # parameters updated in _accumulate_sufficient_statistics.
        for state, g in enumerate(self.gmms_):
            n_features = g.means_.shape[1]
            norm = stats['norm'][state]
            if 'w' in params:
                g.weights_ = normalize(norm.copy())
            if 'm' in params:
                g.means_ = stats['means'][state] / norm[:, np.newaxis]
            if 'c' in params:
                if g.covariance_type == 'tied':
                    g.covars_ = ((stats['covars'][state]
                                 + self.covars_prior * np.eye(n_features))
                                 / norm.sum())
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(g.covars_.ndim)
                    shape[0] = np.shape(g.covars_)[0]
                    cvnorm.shape = shape
                    if g.covariance_type in ['spherical', 'diag']:
                        g.covars_ = (stats['covars'][state] +
                                     self.covars_prior) / cvnorm
                    elif g.covariance_type == 'full':
                        eye = np.eye(n_features)
                        g.covars_ = ((stats['covars'][state]
                                     + self.covars_prior * eye[np.newaxis])
                                     / cvnorm)
