# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
# and Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import string

import numpy as np
from sklearn.utils import check_random_state
from sklearn.mixture import (
    GMM, sample_gaussian,
    distribute_covar_matrix_to_match_covariance_type, _validate_covars)
from sklearn import cluster

from .base import _BaseHMM, decoder_algorithms
from .utils import normalize
from .utils.fixes import log_multivariate_normal_density

__all__ = ['GMMHMM',
           'GaussianHMM',
           'MultinomialHMM',

           # for compatbility, but we should remove this, really.
           'decoder_algorithms',
           'normalize']

NEGINF = -np.inf


class GaussianHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian emissions

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    Attributes
    ----------
    _covariance_type : string
        String describing the type of covariance parameters used by
        the model.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    n_features : int
        Dimensionality of the Gaussian emissions.

    n_components : int
        Number of states in the model.

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    means : array, shape (`n_components`, `n_features`)
        Mean parameters for each state.

    covars : array
        Covariance parameters for each state.  The shape depends on
        ``_covariance_type``::

            (`n_components`,)                   if 'spherical',
            (`n_features`, `n_features`)              if 'tied',
            (`n_components`, `n_features`)           if 'diag',
            (`n_components`, `n_features`, `n_features`)  if 'full'

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means, and 'c' for covars.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means, and 'c' for
        covars.  Defaults to all parameters.


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

    def __init__(self, n_components=1, covariance_type='diag', startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", means_prior=None, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          thresh=thresh, params=params,
                          init_params=init_params)

        self._covariance_type = covariance_type
        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('bad covariance_type')

        self.means_prior = means_prior
        self.means_weight = means_weight

        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    @property
    def covariance_type(self):
        """Covariance type of the model.

        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._covariance_type

    def _get_means(self):
        """Mean parameters for each state."""
        return self._means_

    def _set_means(self, means):
        means = np.asarray(means)
        if (hasattr(self, 'n_features')
                and means.shape != (self.n_components, self.n_features)):
            raise ValueError('means must have shape '
                             '(n_components, n_features)')
        self._means_ = means.copy()
        self.n_features = self._means_.shape[1]

    means_ = property(_get_means, _set_means)

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self._covariance_type == 'full':
            return self._covars_
        elif self._covariance_type == 'diag':
            return [np.diag(cov) for cov in self._covars_]
        elif self._covariance_type == 'tied':
            return [self._covars_] * self.n_components
        elif self._covariance_type == 'spherical':
            return [np.eye(self.n_features) * f for f in self._covars_]

    def _set_covars(self, covars):
        covars = np.asarray(covars)
        _validate_covars(covars, self._covariance_type, self.n_components)
        self._covars_ = covars.copy()

    covars_ = property(_get_covars, _set_covars)

    def _compute_log_likelihood(self, obs):
        return log_multivariate_normal_density(
            obs, self._means_, self._covars_, self._covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        if self._covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self._means_[state], cv, self._covariance_type,
                               random_state=random_state)

    def _init(self, obs, params='stmc'):
        super(GaussianHMM, self)._init(obs, params=params)

        if (hasattr(self, 'n_features')
                and self.n_features != obs[0].shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (obs[0].shape[1],
                                              self.n_features))

        self.n_features = obs[0].shape[1]

        if 'm' in params:
            self._means_ = cluster.KMeans(
                n_clusters=self.n_components).fit(obs[0]).cluster_centers_
        if 'c' in params:
            cv = np.cov(obs[0].T)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self._covariance_type, self.n_components)
            self._covars_[self._covars_ == 0] = 1e-5

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self._covariance_type in ('tied', 'full'):
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
            if self._covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self._covariance_type in ('tied', 'full'):
                for t, o in enumerate(obs):
                    obsobsT = np.outer(o, o)
                    for c in range(self.n_components):
                        stats['obs*obs.T'][c] += posteriors[t, c] * obsobsT

    def _do_mstep(self, stats, params):
        super(GaussianHMM, self)._do_mstep(stats, params)

        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in params:
            prior = self.means_prior
            weight = self.means_weight
            if prior is None:
                weight = 0
                prior = 0
            self._means_ = (weight * prior + stats['obs']) / (weight + denom)

        if 'c' in params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            if covars_prior is None:
                covars_weight = 0
                covars_prior = 0

            means_prior = self.means_prior
            means_weight = self.means_weight
            if means_prior is None:
                means_weight = 0
                means_prior = 0
            meandiff = self._means_ - means_prior

            if self._covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * (meandiff) ** 2
                          + stats['obs**2']
                          - 2 * self._means_ * stats['obs']
                          + self._means_ ** 2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self._covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self._covariance_type in ('tied', 'full'):
                cvnum = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self._means_[c])

                    cvnum[c] = (means_weight * np.outer(meandiff[c],
                                                        meandiff[c])
                                + stats['obs*obs.T'][c]
                                - obsmean - obsmean.T
                                + np.outer(self._means_[c], self._means_[c])
                                * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self._covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cvnum.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self._covariance_type == 'full':
                    self._covars_ = ((covars_prior + cvnum) /
                                     (cvweight + stats['post'][:, None, None]))

    def fit(self, obs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences, each of which
            has shape (n_i, n_features), where n_i is the length of
            the i_th observation.

        Notes
        -----
        In general, `logprob` should be non-decreasing unless
        aggressive pruning is used.  Decreasing `logprob` is generally
        a sign of overfitting (e.g. the covariance parameter on one or
        more components becomminging too small).  You can fix this by getting
        more training data, or increasing covars_prior.
        """
        return super(GaussianHMM, self).fit(obs)


class MultinomialHMM(_BaseHMM):
    """Hidden Markov Model with multinomial (discrete) emissions

    Attributes
    ----------
    n_components : int
        Number of states in the model.

    n_symbols : int
        Number of possible symbols emitted by the model (in the observations).

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    emissionprob : array, shape ('n_components`, 'n_symbols`)
        Probability of emitting a given symbol when in each state.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emmissionprob.
        Defaults to all parameters.

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

    def __init__(self, n_components=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters):
        """Create a hidden Markov model with multinomial emissions.

        Parameters
        ----------
        n_components : int
            Number of states.
        """
        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params)

    def _get_emissionprob(self):
        """Emission probability distribution for each state."""
        return np.exp(self._log_emissionprob)

    def _set_emissionprob(self, emissionprob):
        emissionprob = np.asarray(emissionprob)
        if hasattr(self, 'n_symbols') and \
                emissionprob.shape != (self.n_components, self.n_symbols):
            raise ValueError('emissionprob must have shape '
                             '(n_components, n_symbols)')

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(emissionprob):
            normalize(emissionprob)

        self._log_emissionprob = np.log(emissionprob)
        underflow_idx = np.isnan(self._log_emissionprob)
        self._log_emissionprob[underflow_idx] = NEGINF
        self.n_symbols = self._log_emissionprob.shape[1]

    emissionprob_ = property(_get_emissionprob, _set_emissionprob)

    def _compute_log_likelihood(self, obs):
        return self._log_emissionprob[:, obs].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        rand = random_state.rand()
        symbol = (cdf > rand).argmax()
        return symbol

    def _init(self, obs, params='ste'):
        super(MultinomialHMM, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

        if 'e' in params:
            if not hasattr(self, 'n_symbols'):
                symbols = set()
                for o in obs:
                    symbols = symbols.union(set(o))
                self.n_symbols = len(symbols)
            emissionprob = normalize(self.random_state.rand(self.n_components,
                                                            self.n_symbols), 1)
            self.emissionprob_ = emissionprob

    def _initialize_sufficient_statistics(self):
        stats = super(MultinomialHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_symbols))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MultinomialHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        if 'e' in params:
            for t, symbol in enumerate(obs):
                stats['obs'][:, symbol] += posteriors[t]

    def _do_mstep(self, stats, params):
        super(MultinomialHMM, self)._do_mstep(stats, params)
        if 'e' in params:
            self.emissionprob_ = (stats['obs']
                                  / stats['obs'].sum(1)[:, np.newaxis])

    def _check_input_symbols(self, obs):
        """Check if ``obs`` is a sample from a Multinomial distribution.

        That is ``obs`` should be an array of non-negative integers from
        range ``[min(obs), max(obs)]``, such that each integer from the range
        occurs in ``obs`` at least once.

        For example ``[0, 0, 2, 1, 3, 1, 1]`` is a valid sample from a
        Multinomial distribution, while ``[0, 0, 3, 5, 10]`` is not.
        """
        symbols = np.concatenate(obs)
        if (len(symbols) == 1 or          # not enough data
            symbols.dtype.kind != 'i' or  # not an integer
            np.any(symbols < 0)):         # contains negative integers
            return False

        symbols.sort()
        return np.all(np.diff(symbols) <= 1)

    def fit(self, obs, **kwargs):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        obs : list
            List of array-like observation sequences. Each observation
            sequence should consist of two or more integers from
            range ``[0, n_symbols - 1]``.
        """
        if not self._check_input_symbols(obs):
            raise ValueError("expected a sample from "
                             "a Multinomial distribution.")

        return _BaseHMM.fit(self, obs, **kwargs)


class GMMHMM(_BaseHMM):
    """Hidden Markov Model with Gaussin mixture emissions

    Attributes
    ----------
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

    n_components : int
        Number of states in the model.

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    gmms : array of GMM objects, length `n_components`
        GMM emission distributions for each state.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

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

    def __init__(self, n_components=1, n_mix=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", gmms=None, covariance_type='diag',
                 covars_prior=1e-2, random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        """Create a hidden Markov model with GMM emissions.

        Parameters
        ----------
        n_components : int
            Number of states.
        """
        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params)

        # XXX: Hotfit for n_mix that is incompatible with the scikit's
        # BaseEstimator API
        self.n_mix = n_mix
        self._covariance_type = covariance_type
        self.covars_prior = covars_prior
        self.gmms = gmms
        if gmms is None:
            gmms = []
            for x in range(self.n_components):
                if covariance_type is None:
                    g = GMM(n_mix)
                else:
                    g = GMM(n_mix, covariance_type=covariance_type)
                gmms.append(g)
        self.gmms_ = gmms

    # Read-only properties.
    @property
    def covariance_type(self):
        """Covariance type of the model.

        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._covariance_type

    def _compute_log_likelihood(self, obs):
        return np.array([g.score(obs) for g in self.gmms_]).T

    def _generate_sample_from_state(self, state, random_state=None):
        return self.gmms_[state].sample(1, random_state=random_state).flatten()

    def _init(self, obs, params='stwmc'):
        super(GMMHMM, self)._init(obs, params=params)

        allobs = np.concatenate(obs, 0)
        for g in self.gmms_:
            g.set_params(init_params=params, n_iter=0)
            g.fit(allobs)

    def _initialize_sufficient_statistics(self):
        stats = super(GMMHMM, self)._initialize_sufficient_statistics()
        stats['norm'] = [np.zeros(g.weights_.shape) for g in self.gmms_]
        stats['means'] = [np.zeros(np.shape(g.means_)) for g in self.gmms_]
        stats['covars'] = [np.zeros(np.shape(g.covars_)) for g in self.gmms_]
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(GMMHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        for state, g in enumerate(self.gmms_):
            _, lgmm_posteriors = g.score_samples(obs)
            lgmm_posteriors += np.log(posteriors[:, state][:, np.newaxis]
                                      + np.finfo(np.float).eps)
            gmm_posteriors = np.exp(lgmm_posteriors)
            tmp_gmm = GMM(g.n_components, covariance_type=g.covariance_type)
            n_features = g.means_.shape[1]
            tmp_gmm._set_covars(
                distribute_covar_matrix_to_match_covariance_type(
                    np.eye(n_features), g.covariance_type,
                    g.n_components))
            norm = tmp_gmm._do_mstep(obs, gmm_posteriors, params)

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
                    if (g.covariance_type in ['spherical', 'diag']):
                        g.covars_ = (stats['covars'][state] +
                                     self.covars_prior) / cvnorm
                    elif g.covariance_type == 'full':
                        eye = np.eye(n_features)
                        g.covars_ = ((stats['covars'][state]
                                     + self.covars_prior * eye[np.newaxis])
                                     / cvnorm)
