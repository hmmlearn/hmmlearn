import logging
import numbers

import numpy as np
from scipy import special
from sklearn import cluster
from sklearn.utils import check_random_state

from . import _kl_divergence as _kl, _utils
from ._emissions import BaseCategoricalHMM, BaseGaussianHMM
from .base import VariationalBaseHMM
from .hmm import COVARIANCE_TYPES
from .utils import fill_covars


_log = logging.getLogger(__name__)


class VariationalCategoricalHMM(BaseCategoricalHMM, VariationalBaseHMM):
    """
    Hidden Markov Model with categorical (discrete) emissions trained
    using Variational Inference.

    References:
        * https://cse.buffalo.edu/faculty/mbeal/thesis/

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_prior_ : array, shape (n_components, )
        Prior for the initial state occupation distribution.

    startprob_posterior_ : array, shape (n_components, )
        Posterior estimate of the state occupation distribution.

    transmat_prior_ : array, shape (n_components, n_components)
        Prior for the matrix of transition probabilities between states.

    transmat_posterior_ : array, shape (n_components, n_components)
        Posterior estimate of the transition probabilities between states.

    emissionprob_prior_ : array, shape (n_components, n_features)
        Prior estimatate of emitting a given symbol when in each state.

    emissionprob_posterior_ : array, shape (n_components, n_features)
        Posterior estimate of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.hmm import VariationalCategoricalHMM
    >>> VariationalCategoricalHMM(n_components=2)  #doctest: +ELLIPSIS
    VariationalCategoricalHMM(algorithm='viterbi',...
    """

    def __init__(self, n_components=1,
                 startprob_prior=None, transmat_prior=None,
                 emissionprob_prior=None, n_features=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=100, tol=1e-6, verbose=False,
                 params="ste", init_params="ste",
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

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

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
        super().__init__(
            n_components=n_components, startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm, random_state=random_state,
            n_iter=n_iter, tol=tol, verbose=verbose,
            params=params, init_params=init_params,
            implementation=implementation
        )
        self.emissionprob_prior = emissionprob_prior
        self.n_features = n_features

    def _init(self, X, lengths):
        """
        Initialize model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        super()._init(X, lengths)
        random_state = check_random_state(self.random_state)
        if self._needs_init("e", "emissionprob_posterior_"):
            emissionprob_init = 1 / self.n_features
            if self.emissionprob_prior is not None:
                emissionprob_init = self.emissionprob_prior
            self.emissionprob_prior_ = np.full(
                (self.n_components, self.n_features), emissionprob_init)
            self.emissionprob_posterior_ = random_state.dirichlet(
                alpha=[emissionprob_init] * self.n_features,
                size=self.n_components
            ) * sum(lengths) / self.n_components

    def _estep_begin(self):
        super()._estep_begin()
        # Stored / Computed for efficiency otherwise
        # it would be done in _compute_subnorm_log_likelihood
        self.emissionprob_log_subnorm_ = (
            special.digamma(self.emissionprob_posterior_)
            - special.digamma(
                self.emissionprob_posterior_.sum(axis=1)[:, None]))

    def _check(self):
        """
        Validate model parameters prior to fitting.

        Raises
        ------
        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        super()._check()

        self.emissionprob_prior_ = np.atleast_2d(self.emissionprob_prior_)
        self.emissionprob_posterior_ = \
            np.atleast_2d(self.emissionprob_posterior_)

        if (self.emissionprob_prior_.shape
                != self.emissionprob_posterior_.shape):
            raise ValueError(
                "emissionprob_prior_ and emissionprob_posterior_must"
                "have shape (n_components, n_features)")
        if self.n_features is None:
            self.n_features = self.emissionprob_posterior_.shape[1]
        if (self.emissionprob_posterior_.shape
                != (self.n_components, self.n_features)):
            raise ValueError(
                f"emissionprob_ must have shape"
                f"({self.n_components}, {self.n_features})")

    def _compute_subnorm_log_likelihood(self, X):
        return self.emissionprob_log_subnorm_[:, X.squeeze(1)].T

    def _do_mstep(self, stats):
        """
        Perform the M-step of the VB-EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        super()._do_mstep(stats)
        # emissionprob
        if "e" in self.params:
            self.emissionprob_posterior_ = (
                self.emissionprob_prior_ + stats['obs'])
            # Provide the normalized probabilities at the posterior median
            div = self.emissionprob_posterior_.sum(axis=1)[:, None]
            self.emissionprob_ = self.emissionprob_posterior_ / div

    def _compute_lower_bound(self, log_prob):
        """Compute the lower bound of the model."""
        # First, get the contribution from the state transitions
        # and initial probabilities
        lower_bound = super()._compute_lower_bound(log_prob)

        # The compute the contributions of the emissionprob
        emissionprob_lower_bound = 0
        for i in range(self.n_components):
            emissionprob_lower_bound -= _kl.kl_dirichlet(
                self.emissionprob_posterior_[i], self.emissionprob_prior_[i])
        return lower_bound + emissionprob_lower_bound


class VariationalGaussianHMM(BaseGaussianHMM, VariationalBaseHMM):
    """
    Hidden Markov Model with Multivariate Gaussian Emissions trained
    using Variational Inference.

    References:
        * https://arxiv.org/abs/1605.08618
        * https://core.ac.uk/reader/10883750
        * https://theses.gla.ac.uk/6941/7/2005McGroryPhD.pdf

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_prior_ : array, shape (n_components, )
        Prior for the initial state occupation distribution.

    startprob_posterior_ : array, shape (n_components, )
        Posterior estimate of the state occupation distribution.

    transmat_prior_ : array, shape (n_components, n_components)
        Prior for the matrix of transition probabilities between states.

    transmat_posterior_ : array, shape (n_components, n_components)
        Posterior estimate of the transition probabilities between states.

    means_prior_: array, shape (n_components, n_features)
        Prior estimates for the mean of each state.

    means_posterior_: array, shape (n_components, n_features)
        Posterior estimates for the mean of each state.

    beta_prior_: array, shape (n_components, )
        Prior estimate on the scale of the variance over the means.

    beta_posterior_: array, shape (n_components, )
        Posterior estimate of the scale of the variance over the means.

    covars_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    dof_prior_: int / array
        The Degrees Of Freedom prior for each state's Wishart distribution.
        The type depends on :attr:`covariance_type`:

        * array, shape (n_components, )  if "full",
        * int                            if "tied".

    dof_prior_: int / array
        The Prior on the Degrees Of Freedom
        for each state's Wishart distribution.
        The type depends on :attr:`covariance_type`:

        * array, shape (n_components, )  if "full",
        * int                            if "tied".

    dof_posterior_: int / array
        The Degrees Of Freedom for each state's Wishart distribution.
        The type depends on :attr:`covariance_type`:

        * array, shape (n_components, )  if "full",
        * int                            if "tied".

    scale_prior_ : array
        Prior for the Inverse scale parameter for each state's
        Wishart distribution. The wishart distribution is
        the conjugate prior for the covariance.

        The shape depends on :attr:`covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    scale_posterior_ : array
        Inverse scale parameter for each state's wishart distribution.
        The wishart distribution is the conjugate prior for the covariance.

        The shape depends on :attr:`covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    Examples
    --------
    >>> from hmmlearn.hmm import VariationalGaussianHMM
    >>> VariationalGaussianHMM(n_components=2)  #doctest: +ELLIPSIS
    VariationalGaussianHMM(algorithm='viterbi',...
    """

    def __init__(self, n_components=1, covariance_type="full",
                 startprob_prior=None, transmat_prior=None,
                 means_prior=None, beta_prior=None, dof_prior=None,
                 scale_prior=None, algorithm="viterbi",
                 random_state=None, n_iter=100, tol=1e-6, verbose=False,
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

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        means_prior, beta_prior : array, shape (n_components, ), optional
            Mean and precision of the Normal prior distribtion for
            :attr:`means_`.

        scale_prior, dof_prior : array, optional
            Parameters of the prior distribution for the covariance matrix
            :attr:`covars_`.

            If :attr:`covariance_type` is "spherical" or "diag" the prior is
            the inverse gamma distribution, otherwise --- the inverse Wishart
            distribution.

            The shape of the scale_prior array depends on
            :attr:`covariance_type`:

            * (n_components, )                        if "spherical",
            * (n_components, n_features)              if "diag",
            * (n_components, n_features, n_features)  if "full",
            * (n_features, n_features)                if "tied".

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

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
        super().__init__(
            n_components=n_components, startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm, random_state=random_state,
            n_iter=n_iter, tol=tol, verbose=verbose,
            params=params, init_params=init_params,
            implementation=implementation
        )
        self.covariance_type = covariance_type
        self.means_prior = means_prior
        self.beta_prior = beta_prior
        self.dof_prior = dof_prior
        self.scale_prior = scale_prior

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

    @property
    def means_(self):
        """
        Compat for _BaseGaussianHMM.  We return the mean of the
        approximating distribution, which for us is just `means_posterior_`
        """
        return self.means_posterior_

    def _init(self, X, lengths):
        """
        Initialize model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        super()._init(X, lengths)

        X_mean = X.mean(axis=0)
        # Kmeans will be used for initializing both the means
        # and the covariances
        kmeans = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state,
                                n_init=10)  # sklearn >=1.2 compat.
        kmeans.fit(X)
        cluster_counts = np.bincount(kmeans.predict(X))

        if (self._needs_init("m", "means_prior_")
                or self._needs_init("m", "means_posterior_")
                or self._needs_init("m", "beta_prior_")
                or self._needs_init("m", "beta_posterior_")):
            if self.means_prior is None:
                self.means_prior_ = np.full(
                    (self.n_components, self.n_features), X_mean)
            else:
                self.means_prior_ = self.means_prior
            # Initialize to the data means
            self.means_posterior_ = np.copy(kmeans.cluster_centers_)

            if self.beta_prior is None:
                self.beta_prior_ = np.zeros(self.n_components) + 1
            else:
                self.beta_prior_ = self.beta_prior

            # Count of items in each cluster
            self.beta_posterior_ = np.copy(cluster_counts)

        if (self._needs_init("c", "dof_prior_")
                or self._needs_init("c", "dof_posterior_")
                or self._needs_init("c", "scale_prior_")
                or self._needs_init("c", "scale_posterior_")):
            if self.covariance_type in ("full", "diag", "spherical"):
                if self.dof_prior is None:
                    self.dof_prior_ = np.full(
                        (self.n_components,), self.n_features)
                else:
                    self.dof_prior_ = self.dof_prior
                self.dof_posterior_ = np.copy(cluster_counts)

            elif self.covariance_type == "tied":
                if self.dof_prior is None:
                    self.dof_prior_ = self.n_features
                else:
                    self.dof_prior_ = self.dof_prior
                self.dof_posterior_ = cluster_counts.sum()

            # Covariance posterior comes from the estimate of the data
            # We store and update both W_k and scale_posterior_,
            # as they each are used in the EM-like algorithm
            cv = np.cov(X.T) + 1E-3 * np.eye(X.shape[1])
            self.covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()

            if self.covariance_type == "full":
                if self.scale_prior is None:
                    self.scale_prior_ = np.broadcast_to(
                        np.identity(self.n_features) * 1e-3,
                        (self.n_components, self.n_features, self.n_features)
                    )
                else:
                    self.scale_prior_ = self.scale_prior
                self.scale_posterior_ = (
                    self._covars_
                    * np.asarray(self.dof_posterior_)[:, None, None])

            elif self.covariance_type == "tied":
                if self.scale_prior is None:
                    self.scale_prior_ = np.identity(self.n_features) * 1e-3
                else:
                    self.scale_prior_ = self.scale_prior
                self.scale_posterior_ = self._covars_ * self.dof_posterior_

            elif self.covariance_type == "diag":
                if self.scale_prior is None:
                    self.scale_prior_ = np.full(
                        (self.n_components, self.n_features), 1e-3)
                else:
                    self.scale_prior_ = self.scale_prior
                self.scale_posterior_ = np.einsum(
                    "ij,i->ij",self._covars_, self.dof_posterior_)

            elif self.covariance_type == "spherical":
                if self.scale_prior is None:
                    self.scale_prior_ = np.full((self.n_components, ), 1e-3)
                else:
                    self.scale_prior_ = self.scale_prior
                self.scale_posterior_ = (self._covars_.mean(axis=1)
                                         * self.dof_posterior_)

    def _get_n_fit_scalars_per_param(self):
        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError(
                f"{self.covariance_type} is invalid")
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf + nc,
            "c": {
                "full": nc + nc * nf * (nf + 1) // 2,
                "tied": 1 + nf * (nf + 1) // 2,
                "diag": nc + nc * nf,
                "spherical": nc + nc,

            }[self.covariance_type],
        }

    def _check(self):
        """
        Validate model parameters prior to fitting.

        Raises
        ------
        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError(
                f"{self.covariance_type} is invalid")

        means_shape = (self.n_components, self.n_features)

        self.means_prior_ = np.asarray(self.means_prior_, dtype=float)
        self.means_posterior_ = np.asarray(self.means_posterior_, dtype=float)
        if self.means_prior_.shape != means_shape:
            raise ValueError(
                "means_prior_ have shape (n_components, n_features)")
        if self.means_posterior_.shape != means_shape:
            raise ValueError(
                "means_posterior_ must have shape (n_components, n_features)")

        self.beta_prior_ = np.asarray(self.beta_prior_, dtype=float)
        self.beta_posterior_ = np.asarray(self.beta_posterior_, dtype=float)
        if self.beta_prior_.shape != (self.n_components,):
            raise ValueError(
                "beta_prior_ have shape (n_components,)")

        if self.beta_posterior_.shape != (self.n_components,):
            raise ValueError(
                "beta_posterior_ must have shape (n_components,)")

        if self.covariance_type in ("full", "diag", "spherical"):
            self.dof_prior_ = np.asarray(self.dof_prior_, dtype=float)
            self.dof_posterior_ = np.asarray(self.dof_posterior_, dtype=float)
            if self.dof_prior_.shape != (self.n_components,):
                raise ValueError(
                    "dof_prior_ have shape (n_components,)")

            if self.dof_posterior_.shape != (self.n_components,):
                raise ValueError(
                    "dof_posterior_ must have shape (n_components,)")

        elif self.covariance_type == "tied":
            if not isinstance(self.dof_prior_, numbers.Number):
                raise ValueError("dof_prior_ should be numeric")
            if not isinstance(self.dof_posterior_, numbers.Number):
                raise ValueError("dof_posterior_ should be numeric")

        self.scale_prior_ = np.asarray(self.scale_prior_, dtype=float)
        self.scale_posterior_ = np.asarray(self.scale_posterior_, dtype=float)

        expected = None
        if self.covariance_type == "full":
            expected = (self.n_components, self.n_features, self.n_features)
        elif self.covariance_type == "tied":
            expected = (self.n_features, self.n_features)
        elif self.covariance_type == "diag":
            expected = (self.n_components, self.n_features)
        elif self.covariance_type == "spherical":
            expected = (self.n_components, )
        # Now check the W's
        if self.scale_prior_.shape != expected:
            raise ValueError(f"scale_prior_ must have shape {expected}, "
                             f"found {self.scale_prior_.shape}")

        if self.scale_posterior_.shape != expected:
            raise ValueError(f"scale_posterior_ must have shape {expected}, "
                             f"found {self.scale_posterior_.shape}")

    def _compute_subnorm_log_likelihood(self, X):
        # Refer to the Gruhl/Sick paper for the notation
        # In general, things are neater if we pretend the covariance is
        # full / tied.  Or, we could treat each case separately, and reduce
        # the number of operations. That's left for the future :-)
        nf = self.n_features

        term1 = special.digamma(
            .5 * (self.dof_posterior_ - np.arange(0, nf)[:, None])
        ).sum(axis=0)

        scale_posterior_ = self.scale_posterior_
        if self.covariance_type in ("diag", "spherical"):
            scale_posterior_ = fill_covars(self.scale_posterior_,
                    self.covariance_type, self.n_components, self.n_features)
        W_k = np.linalg.inv(scale_posterior_)
        term1 += nf * np.log(2) + _utils.logdet(W_k)
        term1 /= 2.

        # We ignore the constant that is typically excluded in the literature
        # term2 = self.n_features * log(2 * M_PI) / 2
        term2 = 0
        term3 = nf / self.beta_posterior_

        # (X - Means) * W_k * (X-Means)^T * self.dof_posterior_
        delta = (X - self.means_posterior_[:, None])
        # c is the HMM Component
        # i is the length of the sequence X
        # j, k are the n_features
        # output shape is length * number of components
        if self.covariance_type in ("full", "diag", "spherical"):
            dots = np.einsum("cij,cjk,cik,c->ic",
                             delta, W_k, delta, self.dof_posterior_)
        elif self.covariance_type == "tied":
            dots = np.einsum("cij,jk,cik,->ic",
                             delta, W_k, delta, self.dof_posterior_)
        last_term = .5 * (dots + term3)
        lll = term1 - term2 - last_term
        return lll

    def _do_mstep(self, stats):
        """
        Perform the M-step of VB-EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        super()._do_mstep(stats)

        if "m" in self.params:
            self.beta_posterior_ = self.beta_prior_ + stats['post']
            self.means_posterior_ = np.einsum("i,ij->ij", self.beta_prior_,
                                              self.means_prior_)
            self.means_posterior_ += stats['obs']
            self.means_posterior_ /= self.beta_posterior_[:, None]

        if "c" in self.params:
            if self.covariance_type == "full":
                # Update DOF
                self.dof_posterior_ = self.dof_prior_ + stats['post']
                # Update scale
                self.scale_posterior_ = (
                    self.scale_prior_
                    + stats['obs*obs.T']
                    + np.einsum("c,ci,cj->cij",
                                self.beta_prior_,
                                self.means_prior_,
                                self.means_prior_)
                    - np.einsum("c,ci,cj->cij",
                                self.beta_posterior_,
                                self.means_posterior_,
                                self.means_posterior_))
                self._covars_ = (self.scale_posterior_
                                 / self.dof_posterior_[:, None, None])
            elif self.covariance_type == "tied":
                # Update DOF
                self.dof_posterior_ = self.dof_prior_ + stats['post'].sum()
                # Update scale
                self.scale_posterior_ = (
                    self.scale_prior_
                    + stats['obs*obs.T'].sum(axis=0)
                    + np.einsum("c,ci,cj->ij",
                                self.beta_prior_,
                                self.means_prior_,
                                self.means_prior_)
                    - np.einsum("c,ci,cj->ij",
                                self.beta_posterior_,
                                self.means_posterior_,
                                self.means_posterior_))
                self._covars_ = self.scale_posterior_ / self.dof_posterior_
            elif self.covariance_type == "diag":
                # Update DOF
                self.dof_posterior_ = self.dof_prior_ + stats['post']
                # Update scale
                self.scale_posterior_ = (
                    self.scale_prior_
                    + stats['obs**2']
                    + np.einsum("c,ci,ci->ci",
                                self.beta_prior_,
                                self.means_prior_,
                                self.means_prior_)
                    - np.einsum("c,ci,ci->ci",
                                self.beta_posterior_,
                                self.means_posterior_,
                                self.means_posterior_))
                self._covars_ = (self.scale_posterior_
                                 / self.dof_posterior_[:, None])
            elif self.covariance_type == "spherical":
                # Update DOF
                self.dof_posterior_ = self.dof_prior_ + stats['post']
                # Update scale
                term2 = (stats['obs**2']
                         + np.einsum("c,ci,ci->ci",
                                     self.beta_prior_,
                                     self.means_prior_,
                                     self.means_prior_)
                         - np.einsum("c,ci,ci->ci",
                                     self.beta_posterior_,
                                     self.means_posterior_,
                                     self.means_posterior_))
                self.scale_posterior_ = (
                    self.scale_prior_
                    + term2.mean(axis=1))
                self.scale_posterior_ = self.scale_posterior_
                self._covars_ = (self.scale_posterior_
                                 / self.dof_posterior_)

    def _compute_lower_bound(self, log_prob):

        # First, get the contribution from the state transitions
        # and initial probabilities
        lower_bound = super()._compute_lower_bound(log_prob)

        # The compute the contributions of the emissions
        emissions_lower_bound = 0

        # For ease of implementation, pretend everything is shaped like
        # full covariance.
        scale_posterior_ = self.scale_posterior_
        scale_prior_ = self.scale_prior_
        if self.covariance_type != "full":
            scale_posterior_ = fill_covars(self.scale_posterior_,
                    self.covariance_type, self.n_components, self.n_features)
            scale_prior_ = fill_covars(self.scale_prior_,
                    self.covariance_type, self.n_components, self.n_features)

        W_k = np.linalg.inv(scale_posterior_)

        if self.covariance_type != "tied":
            dof = self.dof_posterior_
        else:
            dof = np.repeat(self.dof_posterior_, self.n_components)

        for i in range(self.n_components):
            precision = W_k[i] * dof[i]
            # KL for the normal distributions
            term1 = np.linalg.inv(self.beta_posterior_[i] * precision)
            term2 = np.linalg.inv(self.beta_prior_[i] * precision)
            kln = _kl.kl_multivariate_normal_distribution(
                self.means_posterior_[i], term1,
                self.means_prior_[i], term2,
            )
            emissions_lower_bound -= kln
            # KL for the wishart distributions
            klw = 0.
            if self.covariance_type in ("full", "diag", "spherical"):
                klw = _kl.kl_wishart_distribution(
                    self.dof_posterior_[i], scale_posterior_[i],
                    self.dof_prior_[i], scale_prior_[i])
            elif self.covariance_type == "tied":
                # Just compute it for the first component
                if i == 0:
                    klw = _kl.kl_wishart_distribution(
                       self.dof_posterior_, self.scale_posterior_,
                       self.dof_prior_, self.scale_prior_)
                else:
                    klw = 0

            emissions_lower_bound -= klw
        return lower_bound + emissions_lower_bound

    def _needs_sufficient_statistics_for_mean(self):
        return 'm' in self.params or 'c' in self.params

    def _needs_sufficient_statistics_for_covars(self):
        return 'c' in self.params
