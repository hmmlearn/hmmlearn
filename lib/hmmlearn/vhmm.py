import logging

import numpy as np
import numpy.linalg

from scipy.special import digamma, logsumexp

from sklearn import cluster
from sklearn.utils import check_array, check_random_state

from . import _hmmc, _utils
from .base import ConvergenceMonitor, DECODER_ALGORITHMS, VariationalBaseHMM
from .kl_divergence import kl_dirichlet, kl_multivariate_normal_distribution, \
        kl_wishart_distribution
from .stats import log_multivariate_normal_density
from .utils import log_mask_zero, log_normalize, normalize


_log = logging.getLogger(__name__)


class VariationalCategoricalHMM(VariationalBaseHMM):
    """
    Hidden Markov Model with categorical (discrete) emissions
    trained using Variational Inference.

    References:
        * https://cse.buffalo.edu/faculty/mbeal/thesis/

    Attributes
    ----------
    n_features_ : int
        Number of possible symbols emitted by the model (in the samples).

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    emissionprob_ : array, shape (n_components, n_features_)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.hmm import VariationalCategoricalHMM
    >>> VariationalCategoricalHMM(n_components=2)  #doctest: +ELLIPSIS
    VariationalCategoricalHMM(algorithm='viterbi',...
    """

    def __init__(self, n_components=1,
                 startprob_prior=None, transmat_prior=None,
                 emissions_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=100, tol=1e-6, verbose=False,
                 params="ste", init_params="ste",
                 implementation="log"):
        super().__init__(
            n_components=n_components, startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm, random_state=random_state,
            n_iter=n_iter, tol=tol, verbose=verbose,
            params=params, init_params=init_params,
            implementation=implementation
        )

        self.emissions_prior = emissions_prior

    def _check_and_set_categorical_features(self, X):
        """
        Check if ``X`` is a sample from a multinomial distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if hasattr(self, "n_features_"):
            if self.n_features_ - 1 < X.max():
                raise ValueError(
                    "Largest symbol is {} but the model only emits symbols up "
                    "to {}".format(X.max(), self.n_features_ - 1))
        else:
            self.n_features_ = X.max() + 1

    def _init(self, X, lengths):
        """
        Initialize model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        super()._init(X, lengths)
        random_state = check_random_state(self.random_state)
        self._check_and_set_categorical_features(X)
        if self._needs_init("e", "emissions_posterior_"):
            emissions_init = 1 / self.n_features_
            if self.emissions_prior is not None:
                emissions_init = self.emissions_prior
            self.emissions_prior_ = np.full(
                (self.n_components, self.n_features_), emissions_init)
            self.emissions_posterior_ = random_state.dirichlet(
                alpha=[emissions_init] * self.n_features_,
                size=self.n_components
            ) * sum(lengths) / self.n_components

    def _update_subnorm(self):
        super()._update_subnorm()
        # Emissions
        self.emissions_log_subnorm_ = (
            digamma(self.emissions_posterior_)
            - digamma(self.emissions_posterior_.sum(axis=1)[:, None])
        )

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
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
        super()._check()

        self.emissions_posterior_ = np.atleast_2d(self.emissions_posterior_)
        n_features_ = getattr(self, "n_features_",
                             self.emissions_posterior_.shape[1])
        if self.emissions_posterior_.shape != (self.n_components, n_features_):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features_)")
        else:
            self.n_features_ = n_features_

    def _compute_subnorm_log_likelihood(self, X):
        return self.emissions_log_subnorm_[:, np.concatenate(X)].T

    def _compute_likelihood(self, X):
        """Computes per-component probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """
        return self.emissionprob_[:, np.concatenate(X)].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def _initialize_sufficient_statistics(self):
        """
        Initialize sufficient statistics required for M-step.

        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.

        Returns
        -------
        nobs : int
            Number of samples in the data.
        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th state.
        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the posterior
            probability of transitioning between the i-th to j-th states.
        """
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features_))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, lattice,
                                          posteriors, fwdlattice, bwdlattice):
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features_)
            Sample sequence.

        lattice : array, shape (n_samples, n_components)
            Probabilities OR Log Probabilities of each sample
            under each of the model states.  Depends on the choice
            of implementation of the Forward-Backward algorithm

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            forward and backward probabilities.
        """
        super()._accumulate_sufficient_statistics(stats=stats, X=X,
                                                  lattice=lattice,
                                                  posteriors=posteriors,
                                                  fwdlattice=fwdlattice,
                                                  bwdlattice=bwdlattice)

        if 'e' in self.params:
            np.add.at(stats['obs'].T, np.concatenate(X), posteriors)

    def _do_mstep(self, stats):
        """
        Perform the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        super()._do_mstep(stats)

        # emissions
        if "e" in self.params:
            self.emissions_posterior_ = self.emissions_prior_ + stats['obs']

            # Provide the normalized probabilities at the posterior median
            self.emissionprob_ = self.emissions_posterior_ / \
                    self.emissions_posterior_.sum(axis=1)[:, None]

    def _lower_bound(self, log_prob):
        """
        Compute the lower bound of the model
        """
        # First, get the contribution from the state transitions
        # and initial probabilities
        lower_bound = super()._lower_bound(log_prob)

        # The compute the contributions of the emissions
        emissions_lower_bound = 0
        for i in range(self.n_components):
            emissions_lower_bound -= kl_dirichlet(
                self.emissions_posterior_[i],
                self.emissions_prior_[i]
            )
        return lower_bound + emissions_lower_bound


class VariationalGaussianHMM(VariationalBaseHMM):
    """
    Hidden Markov Model with Multivariate Gaussian Emissions

    References:
        * https://arxiv.org/abs/1605.08618
        * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.61.3078
        * https://theses.gla.ac.uk/6941/7/2005McGroryPhD.pdf

    Attributes
    ----------
    n_features_ : int
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

    def _init(self, X, lengths):
        """
        Initialize model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        super()._init(X, lengths)
        self.n_features_ = X.shape[1]
        X_mean = X.mean(axis=0)
        # Kmeans will be used for initializing both the means
        # and the covariances
        kmeans = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state)
        kmeans.fit(X)
        cluster_counts = np.bincount(kmeans.predict(X))

        if self._needs_init("m", "means_posterior_"):
            if self.means_prior is None:
                self.means_prior_ = np.full(
                    (self.n_components, self.n_features_), X_mean)
            else:
                self.means_prior_ = self.means_prior
            if self.beta_prior is None:
                self.beta_prior_ = np.zeros(self.n_components) + 1
            else:
                self.beta_prior_ = self.beta_prior_

            # Initialize to the data means
            self.means_posterior_ = np.copy(kmeans.cluster_centers_)
            # Count of items in each cluster
            self.beta_posterior_ = np.copy(cluster_counts)

        if self._needs_init("c", "covars_posterior_"):
            if self.covariance_type == "full":
                if self.dof_prior is None:
                    self.dof_prior_ = np.full(
                        (self.n_components,), self.n_features_)
                else:
                    self.dof_prior_ = self.dof_prior
                if self.scale_prior is None:
                    self.W_0_inv_ = np.broadcast_to(
                        np.identity(self.n_features_) * 1e-3,
                        (self.n_components, self.n_features_, self.n_features_)
                    )
                else:
                    self.W_0_inv_ = self.scale_prior

            elif self.covariance_type == "tied":
                assert False, "Not Finished"
            elif self.covariance_type == "diag":
                assert False, "Not Finished"
            elif self.covariance_type == "spherical":
                assert False, "Not Finished"
            else:
                raise ValueError(
                    f"Unknown covariance_type: {self.covariance_type}")

            # Covariance posterior comes from the estimate of the data
            cv = np.cov(X.T) + 1E-3 * np.eye(X.shape[1])
            self.covars_posterior_ = \
                np.copy(
                    _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components
                    )
                )

            self.dof_posterior_ = np.copy(cluster_counts)
            self.W_k_inv_ = self.covars_posterior_ * \
                    self.dof_posterior_[:, None, None]

            self.W_k_ = np.linalg.inv(self.W_k_inv_)
            self.W_0_ = np.linalg.inv(self.W_0_inv_)

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

    def _compute_subnorm_log_likelihood(self, X):
        # Refer to the Gruhl/Sick paper
        term1 = np.zeros_like(self.dof_posterior_, dtype=float)
        for d in range(1, self.n_features_+1):
            term1 += digamma(.5 * self.dof_posterior_ + 1 - d)
        term1 += self.n_features_ * np.log(2)
        term1 += np.log(numpy.linalg.det(self.W_k_))
        term1 /= 2

        # A constant, typically excluded in the literature
        # self.n_features_ * log(2 * M_PI ) / 2
        term2 = 0
        term3 = self.n_features_ / self.beta_posterior_

        # (X - Means) * W_k * (X-Means)^T * self.dof_posterior_
        delta = (X - self.means_posterior_[:, None])
        # c is the HMM Component
        # i is the length of the sequence X
        # j, k are the n_features_
        # output shape is length * number of components
        dots = np.einsum("cij,cjk,cik,c->ic",
                         delta, self.W_k_, delta, self.dof_posterior_)
        last_term = .5 * (dots + term3)
        lll = term1 - term2 - last_term
        return lll

    def _compute_log_likelihood(self, X):
        """Computes per-component probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """
        return log_multivariate_normal_density(
            X, self.means_posterior_, self.covars_posterior_,
            self.covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.multivariate_normal(
            self.means_posterior_[state], self.covars_posterior_[state]
        )

    def _initialize_sufficient_statistics(self):
        """
        Initialize sufficient statistics required for M-step.

        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.

        Returns
        -------
        nobs : int
            Number of samples in the data.
        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th state.
        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the posterior
            probability of transitioning between the i-th to j-th states.
        """
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features_))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features_))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features_,
                                           self.n_features_))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, lattice,
                                          posteriors, fwdlattice, bwdlattice):
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features_)
            Sample sequence.

        lattice : array, shape (n_samples, n_components)
            Probabilities OR Log Probabilities of each sample
            under each of the model states.  Depends on the choice
            of implementation of the Forward-Backward algorithm

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            forward and backward probabilities.
        """
        super()._accumulate_sufficient_statistics(stats=stats, X=X,
                                                  lattice=lattice,
                                                  posteriors=posteriors,
                                                  fwdlattice=fwdlattice,
                                                  bwdlattice=bwdlattice)
        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, X)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, X ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, X, X)

    def _do_mstep(self, stats):
        """
        Perform the M-step of EM algorithm.

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

            self.dof_posterior_ = self.dof_prior_ + stats['post']
            if self.covariance_type == "full":
                tmp1 = np.einsum("c,ci, cj->cij", self.beta_prior_,
                                 self.means_prior_, self.means_prior_)
                tmp2 = np.einsum("c,ci, cj->cij", self.beta_posterior_,
                                 self.means_posterior_, self.means_posterior_)
                self.W_k_inv_ = self.W_0_inv_ + stats['obs*obs.T'] \
                        + tmp1 - tmp2

                self.W_k_ = np.linalg.inv(self.W_k_inv_)
                self.covars_posterior_ = np.copy(self.W_k_inv_) \
                        / self.dof_posterior_[:, None, None]
            else:
                assert False, "Not finished"

    def _lower_bound(self, log_prob):

        # First, get the contribution from the state transitions
        # and initial probabilities
        lower_bound = super()._lower_bound(log_prob)

        # The compute the contributions of the emissions
        emissions_lower_bound = 0
        for i in range(self.n_components):
            # KL for the normal distributions
            precision = self.W_k_[i] * self.dof_posterior_[i]
            term1 = np.linalg.inv(self.beta_posterior_[i] * precision)
            term2 = np.linalg.inv(self.beta_prior_[i] * precision)
            kln = kl_multivariate_normal_distribution(
                self.means_posterior_[i], term1,
                self.means_prior_[i], term2
            )
            emissions_lower_bound -= kln
            # KL for the wishart distributions
            klw = 0.
            if self.covariance_type == "full":
                klw = kl_wishart_distribution(
                    self.dof_posterior_[i], self.W_k_inv_[i],
                    self.dof_prior_[i], self.W_0_inv_[i])
            else:
                assert False, "Not finished"

            emissions_lower_bound -= klw
        return lower_bound + emissions_lower_bound
