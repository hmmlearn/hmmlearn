"""
The :mod:`hmmlearn.hsmm` module implements hidden semi-Markov models.
"""

import logging

import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson as poisson_dist
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from . import _hmmc, _utils
from .base import BaseHMM
from .hmm import GaussianHMM
from .utils import normalize, log_normalize


__all__ = ["GaussianHSMM"]

_log = logging.getLogger(__name__)


class GaussianHSMM(GaussianHMM):
    r"""Gaussian Hidden Semi-Markov Model with Poisson duration distributions.

    Extends :class:`~hmmlearn.hmm.GaussianHMM` by replacing the implicit
    geometric state-duration distribution (encoded via self-transitions) with
    explicit per-state Poisson distributions.  Self-transitions are
    structurally forbidden; the model controls how long it occupies each
    state through the duration parameters :attr:`dur_lambda_`.

    The duration probability for state :math:`i` is

    .. math::

        p(d \mid i) = \frac{\text{Poisson}(d;\,\lambda_i)}
                          {\sum_{d'=1}^{D_\max}\text{Poisson}(d';\,\lambda_i)},
        \quad d = 1, \ldots, D_\max.

    The E-step reduces the HSMM to an equivalent *augmented* HMM with
    :math:`N \times D_\max` states (:math:`N` = ``n_components``), for which
    the existing Cython forward/backward kernels apply without modification.

    Parameters
    ----------
    n_components : int
        Number of hidden states.
    covariance_type : {"spherical", "diag", "full", "tied"}, optional
        Covariance matrix structure.  Defaults to ``"diag"``.
    min_covar : float, optional
        Floor on diagonal covariances to prevent over-fitting.
    max_duration : int, optional
        Maximum segment duration :math:`D_\max`.  Larger values allow
        the model to represent longer runs but increase the size of the
        augmented state space from :math:`N` to :math:`N \times D_\max`.
        Default is ``10``.
    startprob_prior : float or array, optional
        Dirichlet concentration parameter for :attr:`startprob_`.
    transmat_prior : float or array, optional
        Dirichlet concentration parameter for rows of :attr:`transmat_`.
    means_prior, means_weight : float or array, optional
        Prior mean and precision for Gaussian emissions.
    covars_prior, covars_weight : float or array, optional
        Inverse-gamma / Wishart prior on covariance matrices.
    algorithm : {"viterbi", "map"}, optional
        Decoder used in :meth:`predict` / :meth:`decode`.
    random_state : int or RandomState, optional
        Seed for random number generation.
    n_iter : int, optional
        Maximum EM iterations.
    tol : float, optional
        EM convergence threshold on log-likelihood improvement.
    verbose : bool, optional
        Print per-iteration convergence information.
    params : str, optional
        Which parameters to update at each M-step.  Any combination of
        ``"s"`` (startprob), ``"t"`` (transmat), ``"m"`` (means),
        ``"c"`` (covars), ``"d"`` (duration rates).
    init_params : str, optional
        Which parameters to initialise before fitting.

    Attributes
    ----------
    dur_lambda_ : array, shape (n_components,)
        Estimated Poisson rate parameters for each state's duration.
    monitor_ : ConvergenceMonitor
        EM convergence monitor.

    References
    ----------
    Yu, S.-Z. (2010). Hidden semi-Markov models.
    *Artificial Intelligence*, 174(2), 215–243.

    Examples
    --------
    >>> import numpy as np
    >>> from hmmlearn.hsmm import GaussianHSMM
    >>> model = GaussianHSMM(n_components=2, max_duration=8, n_iter=20,
    ...                      random_state=0)
    >>> X = np.concatenate([np.zeros((40, 1)), np.ones((40, 1))])
    >>> model.fit(X)  # doctest: +ELLIPSIS
    GaussianHSMM(...)
    >>> model.predict(X)  # doctest: +SKIP
    array([0, 0, ..., 1, 1])
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="diag",
        min_covar=1e-3,
        max_duration=10,
        startprob_prior=1.0,
        transmat_prior=1.0,
        means_prior=0,
        means_weight=0,
        covars_prior=1e-2,
        covars_weight=1,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="stmcd",
        init_params="stmcd",
        implementation="log",
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            min_covar=min_covar,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            means_prior=means_prior,
            means_weight=means_weight,
            covars_prior=covars_prior,
            covars_weight=covars_weight,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )
        self.max_duration = max_duration

    # ------------------------------------------------------------------
    # Initialisation and validation
    # ------------------------------------------------------------------

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        if self._needs_init("d", "dur_lambda_"):
            self.dur_lambda_ = np.full(self.n_components,
                                       max(1.0, self.max_duration / 2.0))
        # Enforce no self-transitions.
        np.fill_diagonal(self.transmat_, 0.0)
        normalize(self.transmat_, axis=1)

    def _check(self):
        super()._check()
        self.dur_lambda_ = np.asarray(self.dur_lambda_)
        if self.dur_lambda_.shape != (self.n_components,):
            raise ValueError(
                "dur_lambda_ must have shape (n_components,); "
                f"got {self.dur_lambda_.shape}")
        if not np.all(self.dur_lambda_ > 0):
            raise ValueError("All dur_lambda_ values must be positive.")
        if self.max_duration < 1:
            raise ValueError("max_duration must be at least 1.")
        np.fill_diagonal(self.transmat_, 0.0)
        normalize(self.transmat_, axis=1)

    def _get_n_fit_scalars_per_param(self):
        d = super()._get_n_fit_scalars_per_param()
        d["d"] = self.n_components
        return d

    # ------------------------------------------------------------------
    # Duration helpers
    # ------------------------------------------------------------------

    def _dur_probs(self):
        """Return normalised Poisson PMF on {1,...,D_max} for each state.

        Returns
        -------
        probs : array, shape (n_components, max_duration)
            ``probs[i, d-1]`` = P(duration = d | state = i).
        """
        D = self.max_duration
        d_vals = np.arange(1, D + 1)            # (D,)
        log_pmf = np.array([
            poisson_dist.logpmf(d_vals, lam)
            for lam in self.dur_lambda_
        ])                                       # (N, D)
        log_pmf -= logsumexp(log_pmf, axis=1, keepdims=True)
        with np.errstate(under="ignore"):
            return np.exp(log_pmf)               # (N, D)

    # ------------------------------------------------------------------
    # Augmented-model construction
    # ------------------------------------------------------------------

    def _augment_params(self):
        """Build the (N*D)-state augmented HMM parameters.

        Augmented state ``(i, d)`` at flat index ``i*D + (d-1)`` means
        *in hidden state i with d time steps remaining in the current
        segment*.

        Returns
        -------
        aug_start : array, shape (n_components * max_duration,)
        aug_trans : array, shape (n_components * max_duration,) * 2
        """
        N, D = self.n_components, self.max_duration
        dur_p = self._dur_probs()                # (N, D)

        # Initial distribution: pi[i] * p(d | i)
        aug_start = (self.startprob_[:, None] * dur_p).ravel()
        normalize(aug_start)

        aug_trans = np.zeros((N * D, N * D))

        # Continuation within a run: (i, d) -> (i, d-1) for d >= 2.
        if D >= 2:
            # Vectorised: for each state i, source indices are i*D+1..i*D+D-1,
            # destination indices are i*D+0..i*D+D-2.
            row_idx = np.concatenate([np.arange(i*D + 1, (i+1)*D) for i in range(N)])
            col_idx = np.concatenate([np.arange(i*D,     (i+1)*D - 1) for i in range(N)])
            aug_trans[row_idx, col_idx] = 1.0

        # End of segment: (i, 1) -> (j, d') for j != i.
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Source: i*D + 0 = i*D  (d=1, 0-indexed offset 0)
                # Dest:   j*D .. j*D+D-1 for d'=1..D
                aug_trans[i * D, j * D : j * D + D] = (
                    self.transmat_[i, j] * dur_p[j, :]
                )

        return aug_start, aug_trans

    # ------------------------------------------------------------------
    # EM E-step (override)
    # ------------------------------------------------------------------

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats["dur"] = np.zeros((self.n_components, self.max_duration))
        return stats

    def _do_estep(self, X, lengths):
        """HSMM E-step via the augmented-HMM forward-backward algorithm."""
        N, D = self.n_components, self.max_duration
        aug_start, aug_trans = self._augment_params()

        stats = self._initialize_sufficient_statistics()
        curr_logprob = 0.0

        for sub_X in _utils.split_X_lengths(X, lengths):
            T = sub_X.shape[0]
            log_fp = self._compute_log_likelihood(sub_X)   # (T, N)
            log_fp_aug = np.repeat(log_fp, D, axis=1)      # (T, N*D)

            # Forward / backward on augmented model.
            log_prob, fwd_aug = _hmmc.forward_log(
                aug_start, aug_trans, log_fp_aug)
            bwd_aug = _hmmc.backward_log(
                aug_start, aug_trans, log_fp_aug)

            # Augmented posteriors gamma_aug[t, i*D+(d-1)].
            log_gamma_aug = fwd_aug + bwd_aug
            log_normalize(log_gamma_aug, axis=1)
            with np.errstate(under="ignore"):
                gamma_aug = np.exp(log_gamma_aug)           # (T, N*D)

            # Marginalise over durations to get N-state posteriors.
            gamma = gamma_aug.reshape(T, N, D).sum(axis=2) # (T, N)

            # Augmented xi (summed over time): shape (N*D, N*D).
            if T > 1:
                log_xi_aug = _hmmc.compute_log_xi_sum(
                    fwd_aug, aug_trans, bwd_aug, log_fp_aug)
                with np.errstate(under="ignore"):
                    xi_aug = np.exp(log_xi_aug)
            else:
                xi_aug = np.zeros((N * D, N * D))

            self._hsmm_accumulate(stats, sub_X, gamma, gamma_aug, xi_aug)
            curr_logprob += log_prob

        return stats, curr_logprob

    def _hsmm_accumulate(self, stats, X, gamma, gamma_aug, xi_aug):
        """Accumulate HSMM sufficient statistics for one sub-sequence."""
        N, D = self.n_components, self.max_duration

        stats["nobs"] += 1

        # Start probabilities: posterior at t=0, marginalised over durations.
        if "s" in self.params:
            stats["start"] += gamma[0]

        # Transition statistics: expected segment-boundary transitions (N×N).
        # In the augmented model a segment boundary corresponds to a
        # transition from aug state (i, 1) at time t to (j, d') at time t+1,
        # for j != i.  These entries are in aug_trans rows i*D for each i.
        if "t" in self.params:
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # xi_aug[i*D, j*D : j*D+D] sums over all d'.
                    stats["trans"][i, j] += xi_aug[i * D, j * D : j * D + D].sum()

        # Duration statistics.
        # Expected count of (state i, duration d) segments =
        #   (a) probability of starting the sequence in aug state (i, d), plus
        #   (b) expected number of aug transitions from (j,1) → (i,d) for j≠i.
        if "d" in self.params:
            # (a) initial segment (gamma_aug[0, i*D+(d-1)])
            stats["dur"] += gamma_aug[0].reshape(N, D)
            # (b) subsequent segments via xi_aug
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    stats["dur"][i] += xi_aug[j * D, i * D : i * D + D]

        # Emission statistics (using N-state posteriors).
        if self._needs_sufficient_statistics_for_mean():
            stats["post"] += gamma.sum(axis=0)
            stats["obs"] += gamma.T @ X
        if self._needs_sufficient_statistics_for_covars():
            if self.covariance_type in ("spherical", "diag"):
                stats["obs**2"] += gamma.T @ X ** 2
            elif self.covariance_type in ("tied", "full"):
                stats["obs*obs.T"] += np.einsum("ij,ik,il->jkl", gamma, X, X)

    # ------------------------------------------------------------------
    # EM M-step (override to update dur_lambda_ and enforce no self-loops)
    # ------------------------------------------------------------------

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        # Parent may have placed non-zero mass on the diagonal via the
        # Dirichlet prior; remove it and re-normalise.
        np.fill_diagonal(self.transmat_, 0.0)
        normalize(self.transmat_, axis=1)

        # Update Poisson duration rates: MLE is the expected mean duration.
        if "d" in self.params:
            d_vals = np.arange(1, self.max_duration + 1)   # (D,)
            for i in range(self.n_components):
                total = stats["dur"][i].sum()
                if total > 0:
                    new_lambda = (stats["dur"][i] @ d_vals) / total
                    # Clip to a sensible range.
                    self.dur_lambda_[i] = np.clip(new_lambda, 0.5,
                                                  self.max_duration)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        random_state : int or RandomState, optional
        currstate : int, optional
            Initial state.

        Returns
        -------
        X : array, shape (n_samples, n_features)
        state_sequence : array, shape (n_samples,)
        """
        check_is_fitted(self, "startprob_")
        self._check()

        rng = check_random_state(
            random_state if random_state is not None else self.random_state)

        dur_p = self._dur_probs()                           # (N, D)
        dur_cdf = np.cumsum(dur_p, axis=1)                  # (N, D)
        d_vals = np.arange(1, self.max_duration + 1)

        # No-self-transition CDF for state transitions.
        transmat = self.transmat_.copy()
        np.fill_diagonal(transmat, 0.0)
        normalize(transmat, axis=1)
        trans_cdf = np.cumsum(transmat, axis=1)

        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_)
            currstate = (startprob_cdf > rng.rand()).argmax()

        X_list, states = [], []
        t = 0
        while t < n_samples:
            # Draw duration from the truncated Poisson distribution.
            dur = int(d_vals[(dur_cdf[currstate] > rng.rand()).argmax()])
            dur = min(dur, n_samples - t)
            for _ in range(dur):
                X_list.append(
                    self._generate_sample_from_state(currstate, rng))
                states.append(currstate)
                t += 1
            # Transition to a different state.
            if t < n_samples:
                row = trans_cdf[currstate]
                currstate = (row > rng.rand()).argmax()

        return np.atleast_2d(X_list), np.array(states, dtype=int)

    # ------------------------------------------------------------------
    # Scoring (override to use augmented model)
    # ------------------------------------------------------------------

    def _score_log(self, X, lengths=None, *, compute_posteriors):
        """Log-probability under the HSMM via the augmented forward pass."""
        N, D = self.n_components, self.max_duration
        aug_start, aug_trans = self._augment_params()
        log_prob = 0.0
        sub_posteriors = [np.empty((0, N))]

        for sub_X in _utils.split_X_lengths(X, lengths):
            log_fp = self._compute_log_likelihood(sub_X)
            log_fp_aug = np.repeat(log_fp, D, axis=1)
            log_probij, fwd_aug = _hmmc.forward_log(
                aug_start, aug_trans, log_fp_aug)
            log_prob += log_probij
            if compute_posteriors:
                bwd_aug = _hmmc.backward_log(
                    aug_start, aug_trans, log_fp_aug)
                log_gamma_aug = fwd_aug + bwd_aug
                log_normalize(log_gamma_aug, axis=1)
                with np.errstate(under="ignore"):
                    gamma_aug = np.exp(log_gamma_aug)
                gamma = gamma_aug.reshape(sub_X.shape[0], N, D).sum(axis=2)
                sub_posteriors.append(gamma)

        return log_prob, np.concatenate(sub_posteriors)

    # ------------------------------------------------------------------
    # Decoding (Viterbi via augmented model)
    # ------------------------------------------------------------------

    def _decode_viterbi(self, X):
        """HSMM Viterbi path via augmented model."""
        N, D = self.n_components, self.max_duration
        aug_start, aug_trans = self._augment_params()
        log_fp = self._compute_log_likelihood(X)
        log_fp_aug = np.repeat(log_fp, D, axis=1)
        log_prob, state_seq_aug = _hmmc.viterbi(
            aug_start, aug_trans, log_fp_aug)
        # Augmented index i*D + (d-1) → original state i.
        state_seq = (state_seq_aug // D).astype(int)
        return log_prob, state_seq
