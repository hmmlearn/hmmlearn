# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
#         Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux
# More API changes: Sergei Lebedev <superbobry@gmail.com>

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import functools
import inspect
import logging

import numpy as np
from scipy import linalg, special
from sklearn import cluster
from sklearn.utils import check_random_state

from . import _utils
from .stats import log_multivariate_normal_density
from .base import _BaseHMM
from .utils import fill_covars, log_mask_zero, log_normalize, normalize

__all__ = ["GMMHMM", "GaussianHMM", "MultinomialHMM"]


_log = logging.getLogger(__name__)
COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


def _check_and_set_gaussian_n_features(model, X):
    _, n_features = X.shape
    if hasattr(model, "n_features") and model.n_features != n_features:
        raise ValueError("Unexpected number of dimensions, got {} but "
                         "expected {}".format(n_features, model.n_features))
    model.n_features = n_features


class GaussianHMM(_BaseHMM):
    """
    Hidden Markov Model with Gaussian emissions.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)  #doctest: +ELLIPSIS
    GaussianHMM(algorithm='viterbi',...
    """

    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        covariance_type : {"sperical", "diag", "full", "tied"}, optional
            The type of covariance parameters to use:

            * "spherical" --- each state uses a single variance value that
              applies to all features (default).
            * "diag" --- each state uses a diagonal covariance matrix.
            * "full" --- each state uses a full (i.e. unrestricted)
              covariance matrix.
            * "tied" --- all states use **the same** full covariance matrix.

        min_covar : float, optional
            Floor on the diagonal of the covariance matrix to prevent
            overfitting. Defaults to 1e-3.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        means_prior, means_weight : array, shape (n_components, ), optional
            Mean and precision of the Normal prior distribtion for
            :attr:`means_`.

        covars_prior, covars_weight : array, shape (n_components, ), optional
            Parameters of the prior distribution for the covariance matrix
            :attr:`covars_`.

            If :attr:`covariance_type` is "spherical" or "diag" the prior is
            the inverse gamma distribution, otherwise --- the inverse Wishart
            distribution.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any combination
            of 's' for startprob, 't' for transmat, 'm' for means, and 'c' for
            covars.  Defaults to all parameters.
        """
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    @property
    def covars_(self):
        """Return covars as a full matrix."""
        return fill_covars(self._covars_, self.covariance_type,
                           self.n_components, self.n_features)

    @covars_.setter
    def covars_(self, covars):
        covars = np.array(covars, copy=True)
        _utils._validate_covars(covars, self.covariance_type,
                                self.n_components)
        self._covars_ = covars

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": {
                "spherical": nc,
                "diag": nc * nf,
                "full": nc * nf * (nf + 1) // 2,
                "tied": nf * (nf + 1) // 2,
            }[self.covariance_type],
        }

    def _init(self, X, lengths=None):
        _check_and_set_gaussian_n_features(self, X)
        super()._init(X, lengths=lengths)

        if self._needs_init("m", "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if self._needs_init("c", "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()

    def _check(self):
        super()._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {}'
                             .format(COVARIANCE_TYPES))

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, None]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                c_n = (means_weight * meandiff**2
                       + stats['obs**2']
                       - 2 * self.means_ * stats['obs']
                       + self.means_**2 * denom)
                c_d = max(covars_weight - 1, 0) + denom
                self._covars_ = (covars_prior + c_n) / np.maximum(c_d, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(self._covars_.mean(1)[:, None],
                                            (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                c_n = np.empty((self.n_components, self.n_features,
                                self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])
                    c_n[c] = (means_weight * np.outer(meandiff[c],
                                                      meandiff[c])
                              + stats['obs*obs.T'][c]
                              - obsmean - obsmean.T
                              + np.outer(self.means_[c], self.means_[c])
                              * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + c_n.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self.covariance_type == 'full':
                    self._covars_ = ((covars_prior + c_n) /
                                     (cvweight + stats['post'][:, None, None]))


_MULTINOMIALHMM_DOC_SUFFIX = """

Notes
-----
Unlike other HMM classes, `MultinomialHMM` ``X`` arrays have shape
``(n_samples, 1)`` (instead of ``(n_samples, n_features)``).  Consider using
`sklearn.preprocessing.LabelEncoder` to transform your input to the right
format.
"""


def _multinomialhmm_fix_docstring_shape(func):
    doc = inspect.getdoc(func)
    if doc is None:
        wrapper = func
    else:
        wrapper = functools.wraps(func)(
            lambda *args, **kwargs: func(*args, **kwargs))
        wrapper.__doc__ = (
            doc.replace("(n_samples, n_features)", "(n_samples, 1)")
            + _MULTINOMIALHMM_DOC_SUFFIX)
    return wrapper


class MultinomialHMM(_BaseHMM):
    """
    Hidden Markov Model with multinomial (discrete) emissions.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    emissionprob_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.hmm import MultinomialHMM
    >>> MultinomialHMM(n_components=2)  #doctest: +ELLIPSIS
    MultinomialHMM(algorithm='viterbi',...
    """

    # TODO: accept the prior on emissionprob_ for consistency.
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any
            combination of 's' for startprob, 't' for transmat, and 'e' for
            emissionprob.  Defaults to all parameters.
        """
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

    score_samples, score, decode, predict, predict_proba, sample, fit = map(
        _multinomialhmm_fix_docstring_shape, [
            _BaseHMM.score_samples,
            _BaseHMM.score,
            _BaseHMM.decode,
            _BaseHMM.predict,
            _BaseHMM.predict_proba,
            _BaseHMM.sample,
            _BaseHMM.fit,
        ])

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _init(self, X, lengths=None):
        self._check_and_set_n_features(X)
        super()._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        if 'e' in self.init_params:
            self.emissionprob_ = self.random_state \
                .rand(self.n_components, self.n_features)
            normalize(self.emissionprob_, axis=1)

    def _check(self):
        super()._check()

        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        n_features = getattr(self, "n_features", self.emissionprob_.shape[1])
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features)")
        else:
            self.n_features = n_features

    def _compute_log_likelihood(self, X):
        return log_mask_zero(self.emissionprob_)[:, np.concatenate(X)].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            for t, symbol in enumerate(np.concatenate(X)):
                stats['obs'][:, symbol] += posteriors[t]

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if 'e' in self.params:
            self.emissionprob_ = (
                stats['obs'] / stats['obs'].sum(axis=1, keepdims=True))

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a multinomial distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if hasattr(self, "n_features"):
            if self.n_features - 1 < X.max():
                raise ValueError(
                    "Largest symbol is {} but the model only emits "
                    "symbols up to {}"
                    .format(X.max(), self.n_features - 1))
        self.n_features = X.max() + 1


class GMMHMM(_BaseHMM):
    """
    Hidden Markov Model with Gaussian mixture emissions.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    weights_ : array, shape (n_components, n_mix)
        Mixture weights for each state.

    means_ : array, shape (n_components, n_mix, n_features)
        Mean parameters for each mixture component in each state.

    covars_ : array
        Covariance parameters for each mixture components in each state.

        The shape depends on :attr:`covariance_type`:

        * (n_components, n_mix)                          if "spherical",
        * (n_components, n_mix, n_features)              if "diag",
        * (n_components, n_mix, n_features, n_features)  if "full"
        * (n_components, n_features, n_features)         if "tied".
    """

    def __init__(self, n_components=1, n_mix=1,
                 min_covar=1e-3, startprob_prior=1.0, transmat_prior=1.0,
                 weights_prior=1.0, means_prior=0.0, means_weight=0.0,
                 covars_prior=None, covars_weight=None,
                 algorithm="viterbi", covariance_type="diag",
                 random_state=None, n_iter=10, tol=1e-2,
                 verbose=False, params="stmcw",
                 init_params="stmcw"):
        """
        Parameters
        ----------
        n_components : int
            Number of states in the model.

        n_mix : int
            Number of states in the GMM.

        covariance_type : {"sperical", "diag", "full", "tied"}, optional
            The type of covariance parameters to use:

            * "spherical" --- each state uses a single variance value that
              applies to all features.
            * "diag" --- each state uses a diagonal covariance matrix
              (default).
            * "full" --- each state uses a full (i.e. unrestricted)
              covariance matrix.
            * "tied" --- all mixture components of each state use **the same**
              full covariance matrix (note that this is not the same as for
              `GaussianHMM`).

        min_covar : float, optional
            Floor on the diagonal of the covariance matrix to prevent
            overfitting. Defaults to 1e-3.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        weights_prior : array, shape (n_mix, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`weights_`.

        means_prior, means_weight : array, shape (n_mix, ), optional
            Mean and precision of the Normal prior distribtion for
            :attr:`means_`.

        covars_prior, covars_weight : array, shape (n_mix, ), optional
            Parameters of the prior distribution for the covariance matrix
            :attr:`covars_`.

            If :attr:`covariance_type` is "spherical" or "diag" the prior is
            the inverse gamma distribution, otherwise --- the inverse Wishart
            distribution.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any combination
            of 's' for startprob, 't' for transmat, 'm' for means, 'c'
            for covars, and 'w' for GMM mixing weights.  Defaults to all
            parameters.
        """
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm, random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_mix = n_mix
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nm * nf,
            "c": {
                "spherical": nc * nm,
                "diag": nc * nm * nf,
                "full": nc * nm * nf * (nf + 1) // 2,
                "tied": nc * nf * (nf + 1) // 2,
            }[self.covariance_type],
            "w": nm - 1,
        }

    def _init(self, X, lengths=None):
        _check_and_set_gaussian_n_features(self, X)
        super()._init(X, lengths=lengths)
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        # Default values for covariance prior parameters
        self._init_covar_priors()
        self._fix_priors_shape()

        main_kmeans = cluster.KMeans(n_clusters=nc,
                                     random_state=self.random_state)
        labels = main_kmeans.fit_predict(X)
        kmeanses = []
        for label in range(nc):
            kmeans = cluster.KMeans(n_clusters=nm,
                                    random_state=self.random_state)
            kmeans.fit(X[np.where(labels == label)])
            kmeanses.append(kmeans)

        if self._needs_init("w", "weights_"):
            self.weights_ = np.full((nc, nm), 1 / nm)

        if self._needs_init("m", "means_"):
            self.means_ = np.stack(
                [kmeans.cluster_centers_ for kmeans in kmeanses])

        if self._needs_init("c", "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(nf)
            if not cv.shape:
                cv.shape = (1, 1)
            if self.covariance_type == 'tied':
                self.covars_ = np.zeros((nc, nf, nf))
                self.covars_[:] = cv
            elif self.covariance_type == 'full':
                self.covars_ = np.zeros((nc, nm, nf, nf))
                self.covars_[:] = cv
            elif self.covariance_type == 'diag':
                self.covars_ = np.zeros((nc, nm, nf))
                self.covars_[:] = np.diag(cv)
            elif self.covariance_type == 'spherical':
                self.covars_ = np.zeros((nc, nm))
                self.covars_[:] = cv.mean()

    def _init_covar_priors(self):
        if self.covariance_type == "full":
            if self.covars_prior is None:
                self.covars_prior = 0.0
            if self.covars_weight is None:
                self.covars_weight = -(1.0 + self.n_features + 1.0)
        elif self.covariance_type == "tied":
            if self.covars_prior is None:
                self.covars_prior = 0.0
            if self.covars_weight is None:
                self.covars_weight = -(self.n_mix + self.n_features + 1.0)
        elif self.covariance_type == "diag":
            if self.covars_prior is None:
                self.covars_prior = -1.5
            if self.covars_weight is None:
                self.covars_weight = 0.0
        elif self.covariance_type == "spherical":
            if self.covars_prior is None:
                self.covars_prior = -(self.n_mix + 2.0) / 2.0
            if self.covars_weight is None:
                self.covars_weight = 0.0

    def _fix_priors_shape(self):
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        # If priors are numbers, this function will make them into a
        # matrix of proper shape
        self.weights_prior = np.broadcast_to(
            self.weights_prior, (nc, nm)).copy()
        self.means_prior = np.broadcast_to(
            self.means_prior, (nc, nm, nf)).copy()
        self.means_weight = np.broadcast_to(
            self.means_weight, (nc, nm)).copy()

        if self.covariance_type == "full":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (nc, nm, nf, nf)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, (nc, nm)).copy()
        elif self.covariance_type == "tied":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (nc, nf, nf)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, nc).copy()
        elif self.covariance_type == "diag":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (nc, nm, nf)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, (nc, nm, nf)).copy()
        elif self.covariance_type == "spherical":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (nc, nm)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, (nc, nm)).copy()

    def _check(self):
        super()._check()
        if not hasattr(self, "n_features"):
            self.n_features = self.means_.shape[2]
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        self._init_covar_priors()
        self._fix_priors_shape()

        # Checking covariance type
        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError("covariance_type must be one of {}"
                             .format(COVARIANCE_TYPES))

        self.weights_ = np.array(self.weights_)
        # Checking mixture weights' shape
        if self.weights_.shape != (nc, nm):
            raise ValueError("mixture weights must have shape "
                             "(n_components, n_mix), actual shape: {}"
                             .format(self.weights_.shape))

        # Checking mixture weights' mathematical correctness
        if not np.allclose(self.weights_.sum(axis=1), 1):
            raise ValueError("mixture weights must sum up to 1")

        # Checking means' shape
        self.means_ = np.array(self.means_)
        if self.means_.shape != (nc, nm, nf):
            raise ValueError("mixture means must have shape "
                             "(n_components, n_mix, n_features), "
                             "actual shape: {}".format(self.means_.shape))

        # Checking covariances' shape
        self.covars_ = np.array(self.covars_)
        covars_shape = self.covars_.shape
        needed_shapes = {
            "spherical": (nc, nm),
            "tied": (nc, nf, nf),
            "diag": (nc, nm, nf),
            "full": (nc, nm, nf, nf),
        }
        needed_shape = needed_shapes[self.covariance_type]
        if covars_shape != needed_shape:
            raise ValueError("{!r} mixture covars must have shape {}, "
                             "actual shape: {}"
                             .format(self.covariance_type,
                                     needed_shape, covars_shape))

        # Checking covariances' mathematical correctness
        if (self.covariance_type == "spherical" or
                self.covariance_type == "diag"):
            if np.any(self.covars_ < 0):
                raise ValueError("{!r} mixture covars must be non-negative"
                                 .format(self.covariance_type))
            if np.any(self.covars_ == 0):
                _log.warning("Degenerate mixture covariance")
        elif self.covariance_type == "tied":
            for i, covar in enumerate(self.covars_):
                if not np.allclose(covar, covar.T):
                    raise ValueError("Covariance of state #{} is not symmetric"
                                     .format(i))
                min_eigvalsh = linalg.eigvalsh(covar).min()
                if min_eigvalsh < 0:
                    raise ValueError("Covariance of state #{} is not positive "
                                     "definite".format(i))
                if min_eigvalsh == 0:
                    _log.warning("Covariance of state #%d has a null "
                                 "eigenvalue.", i)
        elif self.covariance_type == "full":
            for i, mix_covars in enumerate(self.covars_):
                for j, covar in enumerate(mix_covars):
                    if not np.allclose(covar, covar.T):
                        raise ValueError(
                            "Covariance of state #{}, mixture #{} is not "
                            "symmetric".format(i, j))
                    min_eigvalsh = linalg.eigvalsh(covar).min()
                    if min_eigvalsh < 0:
                        raise ValueError(
                            "Covariance of state #{}, mixture #{} is not "
                            "positive definite".format(i, j))
                    if min_eigvalsh == 0:
                        _log.warning("Covariance of state #%d, mixture #%d "
                                     "has a null eigenvalue.", i, j)

    def _generate_sample_from_state(self, state, random_state=None):
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        cur_weights = self.weights_[state]
        i_gauss = random_state.choice(self.n_mix, p=cur_weights)
        if self.covariance_type == 'tied':
            # self.covars_.shape == (n_components, n_features, n_features)
            # shouldn't that be (n_mix, ...)?
            covs = self.covars_
        else:
            covs = self.covars_[:, i_gauss]
            covs = fill_covars(covs, self.covariance_type,
                               self.n_components, self.n_features)
        return random_state.multivariate_normal(
            self.means_[state, i_gauss], covs[state]
        )

    def _compute_log_weighted_gaussian_densities(self, X, i_comp):
        cur_means = self.means_[i_comp]
        cur_covs = self.covars_[i_comp]
        if self.covariance_type == 'spherical':
            cur_covs = cur_covs[:, None]
        log_cur_weights = np.log(self.weights_[i_comp])

        return log_multivariate_normal_density(
            X, cur_means, cur_covs, self.covariance_type
        ) + log_cur_weights

    def _compute_log_likelihood(self, X):
        n_samples, _ = X.shape
        res = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, i)
            with np.errstate(under="ignore"):
                res[:, i] = special.logsumexp(log_denses, axis=1)

        return res

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['n_samples'] = 0
        stats['post_mix_sum'] = np.zeros((self.n_components, self.n_mix))
        stats['post_sum'] = np.zeros(self.n_components)

        # The following statistics are stored in lists so we can
        # accumulate chunks of data for multiple sequences (aka
        # multiple frames) during fitting. The fit(X, lengths) method
        # in the _BaseHMM class will call
        # _accumulate_sufficient_statistics once per sequence in the
        # training samples. Data from all sequences needs to be
        # accumulated and fed into _do_mstep.
        #
        # Suppose fit(X, lengths) is called with M>=1 sequences, where
        # each sequence s=0, ..., M-1 contains L[s] = lengths[s]
        # ordered samples. Then after M calls to
        # _accumulate_sufficient_statistics, one per sequence, we
        # expect each list statistic to contain M items, all arrays,
        # with the following shapes:
        #
        # stat              shape of s-th item
        #
        # post_comp_mix     (L[s], n_components, n_mix)
        # samples           (L[s], n_features)
        # centred           (L[s], n_components, n_mix, n_features)
        #
        # FIXME this encoding requires memory proportional to the
        # number of samples. It would be preferable to rework the
        # calculations in _do_mstep to reduce over the samples axis
        # earlier during _accumulate_sufficient_statistics in order to
        # make memory consumption independent of the number of samples.
        stats['post_comp_mix'] = []
        stats['samples'] = []
        stats['centered'] = []
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          post_comp, fwdlattice, bwdlattice):

        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, post_comp, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape

        stats['n_samples'] += n_samples
        stats['samples'].append(X)

        post_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for p in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, p)
            log_normalize(log_denses, axis=-1)
            with np.errstate(under="ignore"):
                post_mix[:, p, :] = np.exp(log_denses)

        with np.errstate(under="ignore"):
            post_comp_mix = post_comp[:, :, None] * post_mix
        stats['post_comp_mix'].append(post_comp_mix)

        stats['post_mix_sum'] += post_comp_mix.sum(axis=0)
        stats['post_sum'] += post_comp.sum(axis=0)

        stats['centered'].append(X[:, None, None, :] - self.means_)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        ns = stats['n_samples']
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        # Aggregate post_comp_mix data from multiple sequences
        post_comp_mix = np.vstack(stats['post_comp_mix'])
        assert post_comp_mix.shape == (ns, nc, nm)

        # Aggregate samples data from multiple sequences
        samples = np.vstack(stats['samples'])
        assert samples.shape == (ns, nf)

        # Aggregate centered data from multiple sequences
        centered = np.vstack(stats['centered'])
        assert centered.shape == (ns, nc, nm, nf)

        # Maximizing weights
        if 'w' in self.params:
            alphas_minus_one = self.weights_prior - 1
            w_n = stats['post_mix_sum'] + alphas_minus_one
            w_d = (stats['post_sum'] + alphas_minus_one.sum(axis=1))[:, None]
            self.weights_ = w_n / w_d

        # Maximizing means
        if 'm' in self.params:
            lambdas, mus = self.means_weight, self.means_prior
            m_n = (
                np.einsum('ijk,il->jkl',
                          post_comp_mix, samples)
                + lambdas[:, :, None] * mus
            )
            m_d = stats['post_mix_sum'] + lambdas
            # If a componenent has zero weight, then replace nan (0/0?) means
            # by 0 (0/1).  The actual value is irrelevant as the component will
            # be unused.  This needs to be done before maximizing covariances
            # as nans would otherwise propagate to other components if
            # covariances are tied.
            m_d[(self.weights_ == 0) & (m_n == 0).all(axis=-1)] = 1
            self.means_ = m_n / m_d[:, :, None]

        # Maximizing covariances
        if 'c' in self.params:
            centered_means = self.means_ - mus

            def outer_f(x):  # Outer product over features.
                return x[..., :, None] * x[..., None, :]

            if self.covariance_type == 'full':
                centered_dots = outer_f(centered)
                centered_means_dots = outer_f(centered_means)

                psis_t = np.transpose(self.covars_prior, axes=(0, 1, 3, 2))
                nus = self.covars_weight

                c_n = (
                    np.einsum('ijk,ijklm->jklm', post_comp_mix, centered_dots)
                    + psis_t
                    + lambdas[:, :, None, None] * centered_means_dots
                )
                c_d = (
                    stats['post_mix_sum'] + 1 + nus + nf + 1
                )[:, :, None, None]

            elif self.covariance_type == 'diag':
                centered2 = centered ** 2
                centered_means2 = centered_means ** 2

                alphas = self.covars_prior
                betas = self.covars_weight

                c_n = (
                    np.einsum('ijk,ijkl->jkl', post_comp_mix, centered2)
                    + lambdas[:, :, None] * centered_means2
                    + 2 * betas
                )
                c_d = stats['post_mix_sum'][:, :, None] + 1 + 2 * (alphas + 1)

            elif self.covariance_type == 'spherical':
                # Much faster than (x**2).sum(-1).
                def norm_last(x): return np.einsum('...i,...i', x, x)
                centered_norm2 = norm_last(centered)
                centered_means_norm2 = norm_last(centered_means)

                alphas = self.covars_prior
                betas = self.covars_weight

                c_n = (
                    np.einsum('ijk,ijk->jk', post_comp_mix, centered_norm2)
                    + lambdas * centered_means_norm2
                    + 2 * betas
                )
                c_d = nf * (stats['post_mix_sum'] + 1) + 2 * (alphas + 1)

            elif self.covariance_type == 'tied':
                centered_dots = outer_f(centered)
                centered_means_dots = outer_f(centered_means)

                psis_t = np.transpose(self.covars_prior, axes=(0, 2, 1))
                nus = self.covars_weight

                c_n = (
                    np.einsum('ijk,ijklm->jlm', post_comp_mix, centered_dots)
                    + np.einsum('ij,ijkl->ikl', lambdas, centered_means_dots)
                    + psis_t
                )
                c_d = (stats['post_sum'] + nm + nus + nf + 1)[:, None, None]

            self.covars_ = c_n / c_d
