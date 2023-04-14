"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import logging

import numpy as np
from scipy import linalg
from sklearn import cluster
from sklearn.utils import check_random_state

from . import _emissions, _utils
from .base import BaseHMM
from .utils import fill_covars, normalize


__all__ = [
    "GMMHMM", "GaussianHMM", "CategoricalHMM", "MultinomialHMM", "PoissonHMM",
]


_log = logging.getLogger(__name__)
COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


class CategoricalHMM(_emissions.BaseCategoricalHMM, BaseHMM):
    """
    Hidden Markov Model with categorical (discrete) emissions.

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
    >>> from hmmlearn.hmm import CategoricalHMM
    >>> CategoricalHMM(n_components=2)  #doctest: +ELLIPSIS
    CategoricalHMM(algorithm='viterbi',...
    """

    def __init__(self, n_components=1, startprob_prior=1.0,
                 transmat_prior=1.0, *, emissionprob_prior=1.0,
                 n_features=None, algorithm="viterbi",
                 random_state=None, n_iter=10, tol=1e-2,
                 verbose=False, params="ste", init_params="ste",
                 implementation="log"):
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

        emissionprob_prior : array, shape (n_components, n_features), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`emissionprob_`.

        n_features: int, optional
            The number of categorical symbols in the HMM.  Will be inferred
            from the data if not set.

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

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        BaseHMM.__init__(self, n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params,
                         implementation=implementation)
        self.emissionprob_prior = emissionprob_prior
        self.n_features = n_features

    def _init(self, X, lengths=None):
        super()._init(X, lengths)

        self.random_state = check_random_state(self.random_state)

        if self._needs_init('e', 'emissionprob_'):
            self.emissionprob_ = self.random_state.rand(
                self.n_components, self.n_features)
            normalize(self.emissionprob_, axis=1)

    def _check(self):
        super()._check()

        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        if self.n_features is None:
            self.n_features = self.emissionprob_.shape[1]
        if self.emissionprob_.shape != (self.n_components, self.n_features):
            raise ValueError(
                f"emissionprob_ must have shape"
                f"({self.n_components}, {self.n_features})")
        self._check_sum_1("emissionprob_")

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if 'e' in self.params:
            self.emissionprob_ = np.maximum(
                self.emissionprob_prior - 1 + stats['obs'], 0)
            normalize(self.emissionprob_, axis=1)


class GaussianHMM(_emissions.BaseGaussianHMM, BaseHMM):
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
                 params="stmc", init_params="stmc",
                 implementation="log"):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        covariance_type : {"spherical", "diag", "full", "tied"}, optional
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

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        super().__init__(n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior, algorithm=algorithm,
                         random_state=random_state, n_iter=n_iter,
                         tol=tol, params=params, verbose=verbose,
                         init_params=init_params,
                         implementation=implementation)
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

    def _init(self, X, lengths=None):
        super()._init(X, lengths)

        if self._needs_init("m", "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state,
                                    n_init=1)  # sklearn <1.4 backcompat.
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
            raise ValueError(
                f"covariance_type must be one of {COVARIANCE_TYPES}")

    def _needs_sufficient_statistics_for_mean(self):
        return 'm' in self.params

    def _needs_sufficient_statistics_for_covars(self):
        return 'c' in self.params

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


class GMMHMM(_emissions.BaseGMMHMM):
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
                 init_params="stmcw",
                 implementation="log"):
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

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        BaseHMM.__init__(self, n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm, random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params,
                         implementation=implementation)
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_mix = n_mix
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    def _init(self, X, lengths=None):
        super()._init(X, lengths=None)
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        def compute_cv():
            return np.cov(X.T) + self.min_covar * np.eye(nf)

        # Default values for covariance prior parameters
        self._init_covar_priors()
        self._fix_priors_shape()

        main_kmeans = cluster.KMeans(n_clusters=nc,
                                     random_state=self.random_state,
                                     n_init=10)  # sklearn >=1.2 compat.
        cv = None  # covariance matrix
        labels = main_kmeans.fit_predict(X)
        main_centroid = np.mean(main_kmeans.cluster_centers_, axis=0)
        means = []
        for label in range(nc):
            kmeans = cluster.KMeans(n_clusters=nm,
                                    random_state=self.random_state,
                                    n_init=10)  # sklearn >=1.2 compat.
            X_cluster = X[np.where(labels == label)]
            if X_cluster.shape[0] >= nm:
                kmeans.fit(X_cluster)
                means.append(kmeans.cluster_centers_)
            else:
                if cv is None:
                    cv = compute_cv()
                m_cluster = np.random.multivariate_normal(main_centroid,
                                                          cov=cv,
                                                          size=nm)
                means.append(m_cluster)

        if self._needs_init("w", "weights_"):
            self.weights_ = np.full((nc, nm), 1 / nm)

        if self._needs_init("m", "means_"):
            self.means_ = np.stack(means)

        if self._needs_init("c", "covars_"):
            if cv is None:
                cv = compute_cv()
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
            raise ValueError(
                f"covariance_type must be one of {COVARIANCE_TYPES}")

        self.weights_ = np.array(self.weights_)
        # Checking mixture weights' shape
        if self.weights_.shape != (nc, nm):
            raise ValueError(
                f"weights_ must have shape (n_components, n_mix), "
                f"actual shape: {self.weights_.shape}")

        # Checking mixture weights' mathematical correctness
        self._check_sum_1("weights_")

        # Checking means' shape
        self.means_ = np.array(self.means_)
        if self.means_.shape != (nc, nm, nf):
            raise ValueError(
                f"means_ must have shape (n_components, n_mix, n_features), "
                f"actual shape: {self.means_.shape}")

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
            raise ValueError(
                f"{self.covariance_type!r} mixture covars must have shape "
                f"{needed_shape}, actual shape: {covars_shape}")

        # Checking covariances' mathematical correctness
        if (self.covariance_type == "spherical" or
                self.covariance_type == "diag"):
            if np.any(self.covars_ < 0):
                raise ValueError(f"{self.covariance_type!r} mixture covars "
                                 f"must be non-negative")
            if np.any(self.covars_ == 0):
                _log.warning("Degenerate mixture covariance")
        elif self.covariance_type == "tied":
            for i, covar in enumerate(self.covars_):
                if not np.allclose(covar, covar.T):
                    raise ValueError(
                        f"Covariance of state #{i} is not symmetric")
                min_eigvalsh = linalg.eigvalsh(covar).min()
                if min_eigvalsh < 0:
                    raise ValueError(
                        f"Covariance of state #{i} is not positive definite")
                if min_eigvalsh == 0:
                    _log.warning("Covariance of state #%d has a null "
                                 "eigenvalue.", i)
        elif self.covariance_type == "full":
            for i, mix_covars in enumerate(self.covars_):
                for j, covar in enumerate(mix_covars):
                    if not np.allclose(covar, covar.T):
                        raise ValueError(
                            f"Covariance of state #{i}, mixture #{j} is not "
                            f"symmetric")
                    min_eigvalsh = linalg.eigvalsh(covar).min()
                    if min_eigvalsh < 0:
                        raise ValueError(
                            f"Covariance of state #{i}, mixture #{j} is not "
                            f"positive definite")
                    if min_eigvalsh == 0:
                        _log.warning("Covariance of state #%d, mixture #%d "
                                     "has a null eigenvalue.", i, j)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        nf = self.n_features
        nm = self.n_mix

        # Maximizing weights
        if 'w' in self.params:
            alphas_minus_one = self.weights_prior - 1
            w_n = stats['post_mix_sum'] + alphas_minus_one
            w_d = (stats['post_sum'] + alphas_minus_one.sum(axis=1))[:, None]
            self.weights_ = w_n / w_d

        # Maximizing means
        if 'm' in self.params:
            m_n = stats['m_n']
            m_d = stats['post_mix_sum'] + self.means_weight
            # If a componenent has zero weight, then replace nan (0/0?) means
            # by 0 (0/1).  The actual value is irrelevant as the component will
            # be unused.  This needs to be done before maximizing covariances
            # as nans would otherwise propagate to other components if
            # covariances are tied.
            m_d[(self.weights_ == 0) & (m_n == 0).all(axis=-1)] = 1
            self.means_ = m_n / m_d[:, :, None]

        # Maximizing covariances
        if 'c' in self.params:
            lambdas, mus = self.means_weight, self.means_prior
            centered_means = self.means_ - mus

            def outer_f(x):  # Outer product over features.
                return x[..., :, None] * x[..., None, :]

            if self.covariance_type == 'full':
                centered_means_dots = outer_f(centered_means)

                psis_t = np.transpose(self.covars_prior, axes=(0, 1, 3, 2))
                nus = self.covars_weight

                c_n = psis_t + lambdas[:, :, None, None] * centered_means_dots
                c_n += stats['c_n']
                c_d = (
                    stats['post_mix_sum'] + 1 + nus + nf + 1
                )[:, :, None, None]

            elif self.covariance_type == 'diag':
                alphas = self.covars_prior
                betas = self.covars_weight
                centered_means2 = centered_means ** 2

                c_n = lambdas[:, :, None] * centered_means2 + 2 * betas
                c_n += stats['c_n']
                c_d = stats['post_mix_sum'][:, :, None] + 1 + 2 * (alphas + 1)

            elif self.covariance_type == 'spherical':
                centered_means_norm2 = np.einsum(  # Faster than (x**2).sum(-1)
                    '...i,...i', centered_means, centered_means)

                alphas = self.covars_prior
                betas = self.covars_weight

                c_n = lambdas * centered_means_norm2 + 2 * betas
                c_n += stats['c_n']
                c_d = nf * (stats['post_mix_sum'] + 1) + 2 * (alphas + 1)

            elif self.covariance_type == 'tied':
                centered_means_dots = outer_f(centered_means)

                psis_t = np.transpose(self.covars_prior, axes=(0, 2, 1))
                nus = self.covars_weight

                c_n = np.einsum('ij,ijkl->ikl',
                                lambdas, centered_means_dots) + psis_t
                c_n += stats['c_n']
                c_d = (stats['post_sum'] + nm + nus + nf + 1)[:, None, None]

            self.covars_ = c_n / c_d


class MultinomialHMM(_emissions.BaseMultinomialHMM):
    """
    Hidden Markov Model with multinomial emissions.

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
    """

    def __init__(self, n_components=1, n_trials=None,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste",
                 implementation="log"):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        n_trials : int or array of int
            Number of trials (when sampling, all samples must have the same
            :attr:`n_trials`).

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

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        BaseHMM.__init__(self, n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params,
                         implementation=implementation)
        self.n_trials = n_trials

        _log.warning(
            "MultinomialHMM has undergone major changes. "
            "The previous version was implementing a CategoricalHMM "
            "(a special case of MultinomialHMM). "
            "This new implementation follows the standard definition for "
            "a Multinomial distribution (e.g. as in "
            "https://en.wikipedia.org/wiki/Multinomial_distribution). "
            "See these issues for details:\n"
            "https://github.com/hmmlearn/hmmlearn/issues/335\n"
            "https://github.com/hmmlearn/hmmlearn/issues/340")

    def _init(self, X, lengths=None):
        super()._init(X, lengths=None)
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
        if self.n_trials is None:
            raise ValueError("n_trials must be set")

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if 'e' in self.params:
            self.emissionprob_ = (
                stats['obs'] / stats['obs'].sum(axis=1, keepdims=True))


class PoissonHMM(_emissions.BasePoissonHMM):
    """
    Hidden Markov Model with Poisson emissions.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    lambdas_ : array, shape (n_components, n_features)
        The expectation value of the waiting time parameters for each
        feature in a given state.
    """

    def __init__(self, n_components=1, startprob_prior=1.0,
                 transmat_prior=1.0, lambdas_prior=0.0,
                 lambdas_weight=0.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stl", init_params="stl",
                 implementation="log"):
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

        lambdas_prior, lambdas_weight : array, shape (n_components,), optional
            The gamma prior on the lambda values using alpha-beta notation,
            respectivley. If None, will be set based on the method of
            moments.

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
            combination of 's' for startprob, 't' for transmat, and 'l' for
            lambdas.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.
        """
        BaseHMM.__init__(self, n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params,
                         implementation=implementation)
        self.lambdas_prior = lambdas_prior
        self.lambdas_weight = lambdas_weight

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        self.random_state = check_random_state(self.random_state)

        mean_X = X.mean()
        var_X = X.var()

        if self._needs_init("l", "lambdas_"):
            # initialize with method of moments based on X
            self.lambdas_ = self.random_state.gamma(
                shape=mean_X**2 / var_X,
                scale=var_X / mean_X,  # numpy uses theta = 1 / beta
                size=(self.n_components, self.n_features))

    def _check(self):
        super()._check()

        self.lambdas_ = np.atleast_2d(self.lambdas_)
        n_features = getattr(self, "n_features", self.lambdas_.shape[1])
        if self.lambdas_.shape != (self.n_components, n_features):
            raise ValueError(
                "lambdas_ must have shape (n_components, n_features)")
        self.n_features = n_features

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        if 'l' in self.params:
            # Based on: Hyvönen & Tolonen, "Bayesian Inference 2019"
            # section 3.2
            # https://vioshyvo.github.io/Bayesian_inference
            alphas, betas = self.lambdas_prior, self.lambdas_weight
            n = stats['post'].sum()
            y_bar = stats['obs'] / stats['post'][:, None]
            # the same as kappa notation (more intuitive) but avoids divide by
            # 0, where:
            # kappas = betas / (betas + n)
            # self.lambdas_ = kappas * (alphas / betas) + (1 - kappas) * y_bar
            self.lambdas_ = (alphas + n * y_bar) / (betas + n)
