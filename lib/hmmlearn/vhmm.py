import logging

import numpy as np
import numpy.linalg

from scipy.special import digamma, logsumexp

from sklearn import cluster
from sklearn.utils import check_array, check_random_state

from . import _hmmc, _utils
from .base import ConvergenceMonitor, DECODER_ALGORITHMS
from .kl_divergence import kl_dirichlet, kl_multivariate_normal_distribution, \
        kl_wishart_distribution
from .utils import log_mask_zero, log_normalize, normalize


_log = logging.getLogger(__name__)


class _VariationalBaseHMM:

    # TODO: accept the prior on emissionprob_ for consistency.
    def __init__(self, n_components=1,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=100, tol=1e-6, verbose=False,
                 params="ste", init_params="ste",
                 implementation="log"):
        self.n_components = n_components
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.params = params
        self.init_params = init_params
        self.implementation = implementation
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter,
                                           self.verbose, strict=True)

    def score_samples(self, X, lengths=None):
        """
        Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        return self._score(X, lengths, compute_posteriors=True)

    def score(self, X, lengths=None):
        """
        Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        return self._score(X, lengths, compute_posteriors=False)[0]

    def _score(self, X, lengths=None, *, compute_posteriors):
        """
        Helper for `score` and `score_samples`.

        Compute the log probability under the model, as well as posteriors if
        *compute_posteriors* is True (otherwise, an empty array is returned
        for the latter).
        """
        _utils.check_is_fitted(self, "startprob_normalized_")
        self._check()

        X = check_array(X)
        impl = {
            "scaling": self._score_scaling,
            "log": self._score_log,
        }[self.implementation]
        return impl(X=X, lengths=lengths,
                    compute_posteriors=compute_posteriors)

    def _score_log(self, X, lengths=None, *, compute_posteriors):
        """
        Compute the log probability under the model, as well as posteriors if
        *compute_posteriors* is True (otherwise, an empty array is returned
        for the latter).
        """
        logprob = 0
        sub_posteriors = [np.empty((0, self.n_components))]
        for sub_X in _utils.split_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(sub_X)
            logprobij, fwdlattice = self._do_forward_log_pass(
                framelogprob, startprob=self.startprob_normalized_,
                transmat=self.transmat_normalized_
            )
            logprob += logprobij
            if compute_posteriors:
                bwdlattice = self._do_backward_log_pass(
                    framelogprob, startprob=self.startprob_normalized_,
                    transmat=self.transmat_normalized_)
                sub_posteriors.append(
                    self._compute_posteriors_log(fwdlattice, bwdlattice))
        return logprob, np.concatenate(sub_posteriors)

    def _score_scaling(self, X, lengths=None, *, compute_posteriors):
        logprob = 0
        sub_posteriors = [np.empty((0, self.n_components))]
        for sub_X in _utils.split_X_lengths(X, lengths):
            frameprob = self._compute_likelihood(sub_X)
            logprobij, fwdlattice, scaling_factors = \
                self._do_forward_scaling_pass(
                    frameprob,
                    startprob=self.startprob_normalized_,
                    transmat=self.transmat_normalized_,
                )
            logprob += logprobij
            if compute_posteriors:
                bwdlattice = self._do_backward_scaling_pass(
                    frameprob, scaling_factors,
                    startprob=self.startprob_normalized_,
                    transmat=self.transmat_normalized_)
                sub_posteriors.append(
                    self._compute_posteriors_scaling(fwdlattice, bwdlattice))

        return logprob, np.concatenate(sub_posteriors)

    def decode(self, X, lengths=None, algorithm=None):
        """
        Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        algorithm : string
            Decoder algorithm. Must be one of "viterbi" or "map".
            If not given, :attr:`decoder` is used.

        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        _utils.check_is_fitted(self, "startprob_normalized_")
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]

        X = check_array(X)
        logprob = 0
        sub_state_sequences = []
        for sub_X in _utils.split_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            sub_logprob, sub_state_sequence = decoder(sub_X)
            logprob += sub_logprob
            sub_state_sequences.append(sub_state_sequence)

        return logprob, np.concatenate(sub_state_sequences)

    def _decode_viterbi(self, X):
        framelogprob = self._compute_log_likelihood(X)
        return self._do_viterbi_pass(framelogprob)

    def _decode_map(self, X):
        _, posteriors = self.score_samples(X)
        logprob = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return logprob, state_sequence

    def predict(self, X, lengths=None):
        """
        Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """
        _, state_sequence = self.decode(X, lengths)
        return state_sequence

    def predict_proba(self, X, lengths=None):
        """
        Compute the posterior probability for each state in the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample from ``X``.
        """
        _, posteriors = self.score_samples(X, lengths)
        return posteriors

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.
        currstate : int
            Current state, as the initial state of the samples.

        Returns
        -------
        X : array, shape (n_samples, n_features_)
            Feature matrix.
        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.

        Examples
        --------
        ::
            # generate samples continuously
            _, Z = model.sample(n_samples=10)
            X, Z = model.sample(n_samples=10, currstate=Z[-1])
        """
        _utils.check_is_fitted(self, "startprob_normalized_")
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        transmat_cdf = np.cumsum(self.transmat_normalized_, axis=1)

        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_normalized_)
            currstate = (startprob_cdf > random_state.rand()).argmax()

        state_sequence = [currstate]
        X = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (transmat_cdf[currstate] > random_state.rand()) \
                .argmax()
            state_sequence.append(currstate)
            X.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

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
        init = 1. / self.n_components
        # We could consider random initialization here as well
        if self._needs_init("s", "startprob_posterior_") or \
           self._needs_init("s", "startprob_prior_"):
            if self.startprob_prior is None:
                startprob_init = init
            else:
                startprob_init = self.startprob_prior

            self.startprob_prior_ = np.full(self.n_components, startprob_init)
            self.startprob_posterior_ = self.startprob_prior_ * len(lengths)

        if self._needs_init("t", "transmat_posterior_"):
            if self.transmat_prior is None:
                transmat_init = init
            else:
                transmat_init = self.transmat_prior
            self.transmat_prior_ = np.full(
                (self.n_components, self.n_components), transmat_init)
            self.transmat_posterior_ = self.transmat_prior_ * \
                sum(lengths) / self.n_components

    def fit(self, X, lengths=None):
        """
        Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        self._init(X, lengths=lengths)
        # self._check()

        self.monitor_._reset()

        impl = {
            "scaling": self._fit_scaling,
            "log": self._fit_log,
        }[self.implementation]
        for iter in range(self.n_iter):
            self._update_subnorm()
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for sub_X in _utils.split_X_lengths(X, lengths):
                lattice, logprob, posteriors, fwdlattice, bwdlattice = \
                        impl(sub_X)
                # Derived HMM classes will implement the following method to
                # update their probability distributions, so keep
                # a single call to this method for simplicity.
                self._accumulate_sufficient_statistics(
                    stats, sub_X, lattice, posteriors, fwdlattice,
                    bwdlattice)
                curr_logprob += logprob

            # Compute the "free energy" / Variational Lower Bound
            # before updating the parameters
            lower_bound = self._lower_bound(curr_logprob)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)
            self.monitor_.report(lower_bound)
            if self.monitor_.converged:
                break
        self._update_normalized()
        return self

    def _lower_bound(self, log_prob):
        startprob_lower_bound = -kl_dirichlet(self.startprob_posterior_,
                                              self.startprob_prior_)
        transmat_lower_bound = 0
        print("lb: start", startprob_lower_bound)
        for i in range(self.n_components):
            transmat_lower_bound -= kl_dirichlet(
                self.transmat_posterior_[i],
                self.transmat_prior_[i]
            )
        print("lb: transmat", transmat_lower_bound)
        print("lb: log_prob", log_prob)
        return startprob_lower_bound + transmat_lower_bound + log_prob

    def _fit_scaling(self, X):
        frameprob = self._compute_subnorm_likelihood(X)
        logprob, fwdlattice, scaling_factors = \
            self._do_forward_scaling_pass(
                frameprob,
                startprob=self.startprob_subnorm_,
                transmat=self.transmat_subnorm_
            )
        bwdlattice = self._do_backward_scaling_pass(
            frameprob,
            scaling_factors,
            startprob=self.startprob_subnorm_,
            transmat=self.transmat_subnorm_
            )
        posteriors = self._compute_posteriors_scaling(fwdlattice, bwdlattice)
        return frameprob, logprob, posteriors, fwdlattice, bwdlattice

    def _fit_log(self, X):
        framelogprob = self._compute_subnorm_log_likelihood(X)
        logprob, fwdlattice = self._do_forward_log_pass(
            framelogprob,
            startprob=self.startprob_subnorm_,
            transmat=self.transmat_subnorm_
        )
        bwdlattice = self._do_backward_log_pass(
            framelogprob,
            startprob=self.startprob_subnorm_,
            transmat=self.transmat_subnorm_
        )
        posteriors = self._compute_posteriors_log(fwdlattice, bwdlattice)
        return framelogprob, logprob, posteriors, fwdlattice, bwdlattice

    def _do_viterbi_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc.viterbi(
            log_mask_zero(self.startprob_normalized_),
            log_mask_zero(self.transmat_normalized_), framelogprob)
        return logprob, state_sequence

    def _do_forward_scaling_pass(self, frameprob, startprob, transmat):
        n_samples, n_components = frameprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        scaling_factors = np.zeros(n_samples)
        fwdlattice, scaling_factors = _hmmc.forward_scaling(
            np.asarray(startprob),
            np.asarray(transmat),
            frameprob,
        )
        log_prob = -np.sum(np.log(scaling_factors))
        return log_prob, fwdlattice, scaling_factors

    def _do_forward_log_pass(self, framelogprob, startprob, transmat):
        n_samples, n_components = framelogprob.shape
        fwdlattice = _hmmc.forward_log(log_mask_zero(startprob),
                                       log_mask_zero(transmat),
                                       framelogprob)
        with np.errstate(under="ignore"):
            return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_scaling_pass(self, frameprob, scaling_factors, startprob,
                                  transmat):
        n_samples, n_components = frameprob.shape
        bwdlattice = _hmmc.backward_scaling(np.asarray(startprob),
                                            np.asarray(transmat),
                                            frameprob, scaling_factors)
        return bwdlattice

    def _do_backward_log_pass(self, framelogprob, startprob, transmat):
        n_samples, n_components = framelogprob.shape
        bwdlattice = _hmmc.backward_log(log_mask_zero(startprob),
                                        log_mask_zero(transmat),
                                        framelogprob)
        return bwdlattice

    def _compute_posteriors_scaling(self, fwdlattice, bwdlattice):
        posteriors = fwdlattice * bwdlattice
        normalize(posteriors, axis=1)
        return posteriors

    def _compute_posteriors_log(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def _needs_init(self, code, name):
        if code in self.init_params:
            if hasattr(self, name):
                _log.warning(
                    "Even though the %r attribute is set, it will be "
                    "overwritten during initialization because 'init_params' "
                    "contains %r", name, code)
            return True
        if not hasattr(self, name):
            return True
        return False

    def _update_subnorm(self):
        # Update PI
        startprob_log_subnorm = digamma(self.startprob_posterior_) \
            - digamma(self.startprob_posterior_.sum())
        self.startprob_subnorm_ = np.exp(startprob_log_subnorm)

        transmat_log_subnorm = digamma(self.transmat_posterior_) - \
            digamma(self.transmat_posterior_.sum(axis=1)[:, None])
        self.transmat_subnorm_ = np.exp(transmat_log_subnorm)

    def _update_normalized(self):
        self.startprob_normalized_ = self.startprob_posterior_ / \
            self.startprob_posterior_.sum()
        self.transmat_normalized_ = self.transmat_posterior_ / \
            self.transmat_posterior_.sum(axis=1)[:, None]

    def _get_n_fit_scalars_per_param(self):
        """
        Return a mapping of fittable parameter names (as in ``self.params``)
        to the number of corresponding scalar parameters that will actually be
        fitted.

        This is used to detect whether the user did not pass enough data points
        for a non-degenerate fit.
        """

    def _check(self):
        """
        Validate model parameters prior to fitting.

        Raises
        ------
        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        nc = self.n_components

        self.startprob_normalized_ = np.asarray(self.startprob_normalized_)
        if len(self.startprob_normalized_) != nc:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_normalized_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {:.4f})"
                             .format(self.startprob_normalized_.sum()))

        self.transmat_normalized_ = np.asarray(self.transmat_normalized_)
        if self.transmat_normalized_.shape != (nc, nc):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_normalized_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ must sum to 1.0 (got {})"
                             .format(self.transmat_normalized_.sum(axis=1)))

    def _compute_subnorm_likelihood(self, X):
        # prevent recursion
        if self._compute_subnorm_log_likelihood != \
           _VariationalBaseHMM._compute_subnorm_log_likelihood.__get__(self):
            return np.exp(self._compute_subnorm_log_likelihood(X))
        else:
            raise NotImplementedError("Must be overridden in subclass")

    def _compute_subnorm_log_likelihood(self, X):
        # prevent recursion
        if self._compute_subnorm_likelihood != \
           _VariationalBaseHMM._compute_subnorm_likelihood.__get__(self):
            return np.log(self._compute_subnorm_likelihood(X))
        else:
            raise NotImplementedError("Must be overridden in subclass")

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
        # prevent recursion
        if self._compute_log_likelihood != \
           _VariationalBaseHMM._compute_log_likelihood.__get__(self):
            return np.exp(self._compute_log_likelihood(X))
        else:
            raise NotImplementedError("Must be overridden in subclass")

    def _compute_log_likelihood(self, X):
        """
        Compute per-component emission log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Emission log probability of each sample in ``X`` for each of the
            model states, i.e., ``log(p(X|state))``.
        """
        # prevent recursion
        if self._compute_likelihood != \
           _VariationalBaseHMM._compute_likelihood.__get__(self):
            return np.log(self._compute_likelihood(X))
        else:
            raise NotImplementedError("Must be overridden in subclass")

    # Methods used by self.fit()

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
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components))}

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

        impl = {
            "scaling": self._accumulate_sufficient_statistics_scaling,
            "log": self._accumulate_sufficient_statistics_log,
        }[self.implementation]

        return impl(stats=stats, X=X, lattice=lattice, posteriors=posteriors,
                    fwdlattice=fwdlattice, bwdlattice=bwdlattice)

    def _accumulate_sufficient_statistics_scaling(self, stats, X, lattice,
                                                  posteriors, fwdlattice,
                                                  bwdlattice):
        """
        Implementation of `_accumulate_sufficient_statistics`
        for ``implementation = "log"``.
        """
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = lattice.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            xi_sum = _hmmc.compute_scaling_xi_sum(fwdlattice,
                                                  self.transmat_subnorm_,
                                                  bwdlattice, lattice)
            stats['trans'] += xi_sum

    def _accumulate_sufficient_statistics_log(self, stats, X, lattice,
                                              posteriors, fwdlattice,
                                              bwdlattice):
        """
        Implementation of `_accumulate_sufficient_statistics`
        for ``implementation = "log"``.
        """
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = lattice.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = _hmmc.compute_log_xi_sum(
                fwdlattice, log_mask_zero(self.transmat_subnorm_), bwdlattice,
                lattice)
            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)

    def _do_mstep(self, stats):
        """
        Perform the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        if 's' in self.params:
            self.startprob_posterior_ = self.startprob_prior_ + stats['start']
        if 't' in self.params:
            self.transmat_posterior_ = self.transmat_prior_ + stats['trans']


class VariationalCategoricalHMM(_VariationalBaseHMM):
    """
    Hidden Markov Model with categorical (discrete) emissions.

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

    # TODO: accept the prior on emissionprob_ for consistency.
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
        self.emissions_subnorm_ = np.exp(self.emissions_log_subnorm_)

    def _update_normalized(self):
        super()._update_normalized()
        self.emissions_normalized_ = self.emissions_posterior_ / \
            self.emissions_posterior_.sum(axis=1)[:, None]

    def _get_n_fit_scalars_per_param(self):
        """
        Return a mapping of fittable parameter names (as in ``self.params``)
        to the number of corresponding scalar parameters that will actually be
        fitted.

        This is used to detect whether the user did not pass enough data points
        for a non-degenerate fit.
        """

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

    def _compute_subnorm_likelihood(self, X):
        return self.emissions_subnorm_[:, np.concatenate(X)].T

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
        return self.emissions_normalized_[:, np.concatenate(X)].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissions_normalized_[state, :])
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

    def _lower_bound(self, log_prob):
        lower_bound = super()._lower_bound(log_prob)
        emissions_lower_bound = 0
        for i in range(self.n_components):
            emissions_lower_bound -= kl_dirichlet(
                self.emissions_posterior_[i],
                self.emissions_prior_[i]
            )
        return lower_bound + emissions_lower_bound


class VariationalGaussianHMM(_VariationalBaseHMM):
    """
    Hidden Markov Model with categorical (discrete) emissions.

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
    >>> from hmmlearn.hmm import VariationalGaussianHMM
    >>> VariationalGaussianHMM(n_components=2)  #doctest: +ELLIPSIS
    VariationalGaussianHMM(algorithm='viterbi',...
    """

    # TODO: accept the prior on emissionprob_ for consistency.
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
        # Kmeans will be used for both means and covariances
        kmeans = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state)
        kmeans.fit(X)
        cluster_counts = np.bincount(kmeans.predict(X))
        if self._needs_init("m", "means_posterior_"):
            self.means_prior_ = np.zeros((self.n_components, self.n_features_)) + X_mean if self.means_prior is None else self.means_prior
            self.beta_prior_ = np.zeros(self.n_components) + 1 if self.beta_prior is None else self.beta_prior_

            # Initialize to the data means
            self.means_posterior_ = np.copy(kmeans.cluster_centers_)
            # Count of items in each cluster
            self.beta_posterior_ = np.copy(cluster_counts)

        if self._needs_init("c", "covars_posterior_"):
            if self.covariance_type == "full":
                if self.dof_prior is None:
                    self.dof_prior_ = np.full((self.n_components,), self.n_features_)
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
                self.dof_prior_ = 1 if self.dof_prior is None else self.dof_prior
                self.W_0_inv_ = 1 if self.scale_prior is None else self.scale_prior
            elif self.covariance_type == "diag":
                assert False, "Not Finished"
                self.dof_prior_ = 1 if self.dof_prior is None else self.dof_prior
                self.W_0_inv_ = 1 if self.scale_prior is None else self.scale_prior
            elif self.covariance_type == "spherical":
                assert False, "Not Finished"
                self.dof_prior_ = 1 if self.dof_prior is None else self.dof_prior
                self.W_0_inv_ = 1 if self.scale_prior is None else self.scale_prior
            else:
                raise ValueError(f"Unknown covariance_type: {covariance_type}")

            cv = np.cov(X.T) + 1E-3 * np.eye(X.shape[1])
            self.covars_posterior_ = \
                np.copy(_utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy())
            self.dof_posterior_ = np.copy(cluster_counts)
            self.W_k_inv_ = self.covars_posterior_ * self.dof_posterior_[:, None, None]

            self.W_k_ = np.linalg.inv(self.W_k_inv_)
            self.W_0_ = np.linalg.inv(self.W_0_inv_)

    def _get_n_fit_scalars_per_param(self):
        """
        Return a mapping of fittable parameter names (as in ``self.params``)
        to the number of corresponding scalar parameters that will actually be
        fitted.

        This is used to detect whether the user did not pass enough data points
        for a non-degenerate fit.
        """

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
        ll = np.zeros((X.shape[0], self.n_components))

        # TODO - remove for loop!
        # in the Gruhl/ Sick paper:
        # .5 (\sum_{d=1}^{D} \digamma(\frac{dof_k + 1 - d}{2}) + D \log(2) + \log \det{W_j})
        term1 = np.zeros_like(self.dof_posterior_, dtype=float)
        for d in range(1, self.n_features_+1):
            term1 += digamma(.5 * self.dof_posterior_ + 1 - d)
        term1 += self.n_features_ * np.log(2) + np.log(numpy.linalg.det(self.W_k_))
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
        dots = np.einsum("cij,cjk,cik,c->ic", delta, self.W_k_, delta, self.dof_posterior_)
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
            X, self.means_posterior_, self.covars_posterior_, self.covariance_type)

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

        # emissions
        if "m" in self.params:
            self.beta_posterior_ = self.beta_prior_ + stats['post']
            self.means_posterior_ = np.einsum("i,ij->ij", self.beta_prior_, self.means_prior_)
            self.means_posterior_ += stats['obs']
            self.means_posterior_ /= self.beta_posterior_[:, None]


        if "c" in self.params:

            self.dof_posterior_ = self.dof_prior_ + stats['post']
            if self.covariance_type == "full":

                for state in range(self.n_components):

                    tmp1 = self.beta_prior_[state] * np.outer(self.means_prior_[state], self.means_prior_[state])
                    tmp2 = self.beta_posterior_[state] * np.outer(self.means_posterior_[state], self.means_posterior_[state])
                    tmp3 = self.W_0_inv_[state] + stats['obs*obs.T'][state] + tmp1 - tmp2
                    self.W_k_inv_[state] = tmp3

                self.W_k_ = np.linalg.inv(self.W_k_inv_)
                self.covars_posterior_ = np.copy(self.W_k_inv_) / self.dof_posterior_[:, None, None]
            else:
                assert False, "Not finished"

    def _lower_bound(self, log_prob):
        lower_bound = super()._lower_bound(log_prob)
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
