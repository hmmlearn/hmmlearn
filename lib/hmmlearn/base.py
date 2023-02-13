import logging
import string
import sys
from collections import deque

import numpy as np
from scipy import linalg, special
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    check_array, check_is_fitted, check_random_state)

from . import _hmmc, _kl_divergence as _kl, _utils
from .utils import normalize, log_normalize


_log = logging.getLogger(__name__)
#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map"))


class ConvergenceMonitor:
    """
    Monitor and report convergence to :data:`sys.stderr`.

    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.
    iter : int
        Number of iterations performed while training the model.

    Examples
    --------
    Use custom convergence criteria by subclassing ``ConvergenceMonitor``
    and redefining the ``converged`` method. The resulting subclass can
    be used by creating an instance and pointing a model's ``monitor_``
    attribute to it prior to fitting.

    >>> from hmmlearn.base import ConvergenceMonitor
    >>> from hmmlearn import hmm
    >>>
    >>> class ThresholdMonitor(ConvergenceMonitor):
    ...     @property
    ...     def converged(self):
    ...         return (self.iter == self.n_iter or
    ...                 self.history[-1] >= self.tol)
    >>>
    >>> model = hmm.GaussianHMM(n_components=2, tol=5, verbose=True)
    >>> model.monitor_ = ThresholdMonitor(model.monitor_.tol,
    ...                                   model.monitor_.n_iter,
    ...                                   model.monitor_.verbose)
    """

    _template = "{iter:>10d} {log_prob:>16.8f} {delta:>+16.8f}"

    def __init__(self, tol, n_iter, verbose):
        """
        Parameters
        ----------
        tol : double
            Convergence threshold.  EM has converged either if the maximum
            number of iterations is reached or the log probability improvement
            between the two consecutive iterations is less than threshold.
        n_iter : int
            Maximum number of iterations to perform.
        verbose : bool
            Whether per-iteration convergence reports are printed.
        """
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque()
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = sorted(dict(vars(self), history=list(self.history)).items())
        return ("{}(\n".format(class_name)
                + "".join(map("    {}={},\n".format, *zip(*params)))
                + ")")

    def _reset(self):
        """Reset the monitor's state."""
        self.iter = 0
        self.history.clear()

    def report(self, log_prob):
        """
        Report convergence to :data:`sys.stderr`.

        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.

        Parameters
        ----------
        log_prob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            delta = log_prob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, log_prob=log_prob, delta=delta)
            print(message, file=sys.stderr)

        # Allow for some wiggleroom based on precision.
        precision = np.finfo(float).eps ** (1/2)
        if self.history and (log_prob - self.history[-1]) < -precision:
            delta = log_prob - self.history[-1]
            _log.warning(f"Model is not converging.  Current: {log_prob}"
                         f" is not greater than {self.history[-1]}."
                         f" Delta is {delta}")
        self.history.append(log_prob)
        self.iter += 1

    @property
    def converged(self):
        """Whether the EM algorithm converged."""
        # XXX we might want to check that ``log_prob`` is non-decreasing.
        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and
                 self.history[-1] - self.history[-2] < self.tol))


class _AbstractHMM(BaseEstimator):
    """
    Base class for Hidden Markov Models learned via Expectation-Maximization
    and Variational Bayes.
    """

    def __init__(self, n_components, algorithm, random_state, n_iter,
                 tol, verbose, params, init_params, implementation):
        """
        Parameters
        ----------
        n_components : int
            Number of states in the model.
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
            of 's' for startprob, 't' for transmat, and other characters for
            subclass-specific emission parameters.  Defaults to all parameters.
        implementation: string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.  However, the
            scaling implementation is generally faster.
        """

        self.n_components = n_components
        self.params = params
        self.init_params = init_params
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.implementation = implementation
        self.random_state = random_state

    def score_samples(self, X, lengths=None):
        """
        Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        log_prob : float
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
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        log_prob : float
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
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        impl = {
            "scaling": self._score_scaling,
            "log": self._score_log,
        }[self.implementation]
        return impl(
            X=X, lengths=lengths, compute_posteriors=compute_posteriors)

    def _score_log(self, X, lengths=None, *, compute_posteriors):
        """
        Compute the log probability under the model, as well as posteriors if
        *compute_posteriors* is True (otherwise, an empty array is returned
        for the latter).
        """
        log_prob = 0
        sub_posteriors = [np.empty((0, self.n_components))]
        for sub_X in _utils.split_X_lengths(X, lengths):
            log_frameprob = self._compute_log_likelihood(sub_X)
            log_probij, fwdlattice = _hmmc.forward_log(
                self.startprob_, self.transmat_, log_frameprob)
            log_prob += log_probij
            if compute_posteriors:
                bwdlattice = _hmmc.backward_log(
                    self.startprob_, self.transmat_, log_frameprob)
                sub_posteriors.append(
                    self._compute_posteriors_log(fwdlattice, bwdlattice))
        return log_prob, np.concatenate(sub_posteriors)

    def _score_scaling(self, X, lengths=None, *, compute_posteriors):
        log_prob = 0
        sub_posteriors = [np.empty((0, self.n_components))]
        for sub_X in _utils.split_X_lengths(X, lengths):
            frameprob = self._compute_likelihood(sub_X)
            log_probij, fwdlattice, scaling_factors = _hmmc.forward_scaling(
                self.startprob_, self.transmat_, frameprob)
            log_prob += log_probij
            if compute_posteriors:
                bwdlattice = _hmmc.backward_scaling(
                    self.startprob_, self.transmat_,
                    frameprob, scaling_factors)
                sub_posteriors.append(
                    self._compute_posteriors_scaling(fwdlattice, bwdlattice))

        return log_prob, np.concatenate(sub_posteriors)

    def _decode_viterbi(self, X):
        log_frameprob = self._compute_log_likelihood(X)
        return _hmmc.viterbi(self.startprob_, self.transmat_, log_frameprob)

    def _decode_map(self, X):
        _, posteriors = self.score_samples(X)
        log_prob = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return log_prob, state_sequence

    def decode(self, X, lengths=None, algorithm=None):
        """
        Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        algorithm : string
            Decoder algorithm. Must be one of "viterbi" or "map".
            If not given, :attr:`decoder` is used.

        Returns
        -------
        log_prob : float
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
        check_is_fitted(self, "startprob_")
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError(f"Unknown decoder {algorithm!r}")

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]

        X = check_array(X)
        log_prob = 0
        sub_state_sequences = []
        for sub_X in _utils.split_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            sub_log_prob, sub_state_sequence = decoder(sub_X)
            log_prob += sub_log_prob
            sub_state_sequences.append(sub_state_sequence)

        return log_prob, np.concatenate(sub_state_sequences)

    def predict(self, X, lengths=None):
        """
        Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
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
        X : array-like, shape (n_samples, n_features)
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
        X : array, shape (n_samples, n_features)
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
        check_is_fitted(self, "startprob_")
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_)
            currstate = (startprob_cdf > random_state.rand()).argmax()

        state_sequence = [currstate]
        X = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (
                (transmat_cdf[currstate] > random_state.rand()).argmax())
            state_sequence.append(currstate)
            X.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

    def fit(self, X, lengths=None):
        """
        Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
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

        if lengths is None:
            lengths = np.asarray([X.shape[0]])

        self._init(X, lengths)
        self._check()
        self.monitor_._reset()

        for iter in range(self.n_iter):
            stats, curr_logprob = self._do_estep(X, lengths)

            # Compute lower bound before updating model parameters
            lower_bound = self._compute_lower_bound(curr_logprob)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)
            self.monitor_.report(lower_bound)
            if self.monitor_.converged:
                break

            if (self.transmat_.sum(axis=1) == 0).any():
                _log.warning("Some rows of transmat_ have zero sum because no "
                             "transition from the state was ever observed.")
        return self

    def _fit_scaling(self, X):
        raise NotImplementedError("Must be overridden in subclass")

    def _fit_log(self, X):
        raise NotImplementedError("Must be overridden in subclass")

    def _compute_posteriors_scaling(self, fwdlattice, bwdlattice):
        posteriors = fwdlattice * bwdlattice
        normalize(posteriors, axis=1)
        return posteriors

    def _compute_posteriors_log(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by log_prob at
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

    def _check_and_set_n_features(self, X):
        _, n_features = X.shape
        if hasattr(self, "n_features"):
            if self.n_features != n_features:
                raise ValueError(
                    f"Unexpected number of dimensions, got {n_features} but "
                    f"expected {self.n_features}")
        else:
            self.n_features = n_features

    def _get_n_fit_scalars_per_param(self):
        """
        Return a mapping of fittable parameter names (as in ``self.params``)
        to the number of corresponding scalar parameters that will actually be
        fitted.

        This is used to detect whether the user did not pass enough data points
        for a non-degenerate fit.
        """
        raise NotImplementedError("Must be overridden in subclass")

    def _check_sum_1(self, name):
        """Check that an array describes one or more distributions."""
        s = getattr(self, name).sum(axis=-1)
        if not np.allclose(s, 1):
            raise ValueError(
                f"{name} must sum to 1 (got {s:.4f})" if s.ndim == 0 else
                f"{name} rows must sum to 1 (got {s})" if s.ndim == 1 else
                "Expected 1D or 2D array")

    def _check(self):
        """
        Validate model parameters prior to fitting.

        Raises
        ------
        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        raise NotImplementedError("Must be overridden in subclass")

    def _compute_likelihood(self, X):
        """
        Compute per-component probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """
        if (self._compute_log_likelihood  # prevent recursion
                != __class__._compute_log_likelihood.__get__(self)):
            # Probabilities equal to zero do occur, and exp(-LARGE) = 0 is OK.
            with np.errstate(under="ignore"):
                return np.exp(self._compute_log_likelihood(X))
        else:
            raise NotImplementedError("Must be overridden in subclass")

    def _compute_log_likelihood(self, X):
        """
        Compute per-component emission log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
            Emission log probability of each sample in ``X`` for each of the
            model states, i.e., ``log(p(X|state))``.
        """
        if (self._compute_likelihood  # prevent recursion
                != __class__._compute_likelihood.__get__(self)):
            # Probabilities equal to zero do occur, and log(0) = -inf is OK.
            likelihood = self._compute_likelihood(X)
            with np.errstate(divide="ignore"):
                return np.log(likelihood)
        else:
            raise NotImplementedError("Must be overridden in subclass")

    def _generate_sample_from_state(self, state, random_state):
        """
        Generate a random sample from a given component.

        Parameters
        ----------
        state : int
            Index of the component to condition on.
        random_state: RandomState
            A random number generator instance.  (`sample` is the only caller
            for this method and already normalizes *random_state*.)

        Returns
        -------
        X : array, shape (n_features, )
            A random sample from the emission distribution corresponding
            to a given component.
        """
        return ()

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

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        """
        Update sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~.BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
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

    def _accumulate_sufficient_statistics_scaling(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
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
            xi_sum = _hmmc.compute_scaling_xi_sum(
                fwdlattice, self.transmat_, bwdlattice, lattice)
            stats['trans'] += xi_sum

    def _accumulate_sufficient_statistics_log(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
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
                fwdlattice, self.transmat_, bwdlattice, lattice)
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

    def _do_estep(self, X, lengths):
        impl = {
            "scaling": self._fit_scaling,
            "log": self._fit_log,
        }[self.implementation]

        stats = self._initialize_sufficient_statistics()
        self._estep_begin()
        curr_logprob = 0
        for sub_X in _utils.split_X_lengths(X, lengths):
            lattice, logprob, posteriors, fwdlattice, bwdlattice = impl(sub_X)
            # Derived HMM classes will implement the following method to
            # update their probability distributions, so keep
            # a single call to this method for simplicity.
            self._accumulate_sufficient_statistics(
                stats, sub_X, lattice, posteriors, fwdlattice,
                bwdlattice)
            curr_logprob += logprob
        return stats, curr_logprob

    def _estep_begin(self):
        pass

    def _compute_lower_bound(self, curr_logprob):
        raise NotImplementedError("Must be overridden in subclass")


class BaseHMM(_AbstractHMM):
    """
    Base class for Hidden Markov Models learned from Expectation-Maximization.

    This class allows for easy evaluation of, sampling from, and maximum a
    posteriori estimation of the parameters of a HMM.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    Notes
    -----
    Normally, one should use a subclass of `.BaseHMM`, with its specialization
    towards a given emission model.  In rare cases, the base class can also be
    useful in itself, if one simply wants to generate a sequence of states
    using `.BaseHMM.sample`.  In that case, the feature matrix will have zero
    features.
    """

    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 implementation="log"):
        """
        Parameters
        ----------
        n_components : int
            Number of states in the model.
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
            before (``init_params``) the training.  Can contain any combination
            of 's' for startprob, 't' for transmat, and other characters for
            subclass-specific emission parameters.  Defaults to all parameters.
        implementation: string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.  However, the
            scaling implementation is generally faster.
        """
        super().__init__(
            n_components=n_components, algorithm=algorithm,
            random_state=random_state, n_iter=n_iter, tol=tol,
            verbose=verbose, params=params, init_params=init_params,
            implementation=implementation)
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)

    def get_stationary_distribution(self):
        """Compute the stationary distribution of states."""
        # The stationary distribution is proportional to the left-eigenvector
        # associated with the largest eigenvalue (i.e., 1) of the transition
        # matrix.
        check_is_fitted(self, "transmat_")
        eigvals, eigvecs = linalg.eig(self.transmat_.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    def _fit_scaling(self, X):
        frameprob = self._compute_likelihood(X)
        log_prob, fwdlattice, scaling_factors = _hmmc.forward_scaling(
            self.startprob_, self.transmat_, frameprob)
        bwdlattice = _hmmc.backward_scaling(
            self.startprob_, self.transmat_, frameprob, scaling_factors)
        posteriors = self._compute_posteriors_scaling(fwdlattice, bwdlattice)
        return frameprob, log_prob, posteriors, fwdlattice, bwdlattice

    def _fit_log(self, X):
        log_frameprob = self._compute_log_likelihood(X)
        log_prob, fwdlattice = _hmmc.forward_log(
            self.startprob_, self.transmat_, log_frameprob)
        bwdlattice = _hmmc.backward_log(
            self.startprob_, self.transmat_, log_frameprob)
        posteriors = self._compute_posteriors_log(fwdlattice, bwdlattice)
        return log_frameprob, log_prob, posteriors, fwdlattice, bwdlattice

    def _do_mstep(self, stats):
        """
        Perform the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        # If a prior is < 1, `prior - 1 + starts['start']` can be negative.  In
        # that case maximization of (n1+e1) log p1 + ... + (ns+es) log ps under
        # the conditions sum(p) = 1 and all(p >= 0) show that the negative
        # terms can just be set to zero.
        # The ``np.where`` calls guard against updating forbidden states
        # or transitions in e.g. a left-right HMM.
        if 's' in self.params:
            startprob_ = np.maximum(self.startprob_prior - 1 + stats['start'],
                                    0)
            self.startprob_ = np.where(self.startprob_ == 0, 0, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = np.maximum(self.transmat_prior - 1 + stats['trans'], 0)
            self.transmat_ = np.where(self.transmat_ == 0, 0, transmat_)
            normalize(self.transmat_, axis=1)

    def _compute_lower_bound(self, curr_logprob):
        return curr_logprob

    def _init(self, X, lengths=None):
        """
        Initialize model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        """
        self._check_and_set_n_features(X)
        init = 1. / self.n_components
        random_state = check_random_state(self.random_state)
        if self._needs_init("s", "startprob_"):
            self.startprob_ = random_state.dirichlet(
                np.full(self.n_components, init))
        if self._needs_init("t", "transmat_"):
            self.transmat_ = random_state.dirichlet(
                np.full(self.n_components, init), size=self.n_components)
        n_fit_scalars_per_param = self._get_n_fit_scalars_per_param()
        if n_fit_scalars_per_param is not None:
            n_fit_scalars = sum(
                n_fit_scalars_per_param[p] for p in self.params)
            if X.size < n_fit_scalars:
                _log.warning(
                    "Fitting a model with %d free scalar parameters with only "
                    "%d data points will result in a degenerate solution.",
                    n_fit_scalars, X.size)

    def _check_sum_1(self, name):
        """Check that an array describes one or more distributions."""
        s = getattr(self, name).sum(axis=-1)
        if not np.allclose(s, 1):
            raise ValueError(
                f"{name} must sum to 1 (got {s:.4f})" if s.ndim == 0 else
                f"{name} rows must sum to 1 (got {s})" if s.ndim == 1 else
                "Expected 1D or 2D array")

    def _check(self):
        """
        Validate model parameters prior to fitting.

        Raises
        ------
        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        self._check_sum_1("startprob_")

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        self._check_sum_1("transmat_")

    def aic(self, X, lengths=None):
        """
        Akaike information criterion for the current model on the input X.

        AIC = -2*logLike + 2 * num_free_params

        https://en.wikipedia.org/wiki/Akaike_information_criterion

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        aic : float
            The lower the better.
        """
        n_params = sum(self._get_n_fit_scalars_per_param().values())
        return -2 * self.score(X, lengths=lengths) + 2 * n_params

    def bic(self, X, lengths=None):
        """
        Bayesian information criterion for the current model on the input X.

        BIC = -2*logLike + num_free_params * log(num_of_data)

        https://en.wikipedia.org/wiki/Bayesian_information_criterion

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        bic : float
            The lower the better.
        """
        n_params = sum(self._get_n_fit_scalars_per_param().values())
        return -2 * self.score(X, lengths=lengths) + n_params * np.log(len(X))

_BaseHMM = BaseHMM  # Backcompat name, will be deprecated in the future.


class VariationalBaseHMM(_AbstractHMM):

    def __init__(self, n_components=1,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=100, tol=1e-6, verbose=False,
                 params="ste", init_params="ste",
                 implementation="log"):
        super().__init__(
            n_components=n_components, algorithm=algorithm,
            random_state=random_state, n_iter=n_iter, tol=tol,
            verbose=verbose, params=params, init_params=init_params,
            implementation=implementation)

        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.monitor_ = ConvergenceMonitor(
            self.tol, self.n_iter, self.verbose)

    def _init(self, X, lengths=None):
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
        self._check_and_set_n_features(X)
        nc = self.n_components
        uniform_prior = 1 / nc
        random_state = check_random_state(self.random_state)
        if (self._needs_init("s", "startprob_posterior_")
                or self._needs_init("s", "startprob_prior_")):
            if self.startprob_prior is None:
                startprob_init = uniform_prior
            else:
                startprob_init = self.startprob_prior

            self.startprob_prior_ = np.full(nc, startprob_init)
            self.startprob_posterior_ = random_state.dirichlet(
                np.full(nc, uniform_prior)) * len(lengths)

        if (self._needs_init("t", "transmat_posterior_")
                or self._needs_init("t", "transmat_prior_")):
            if self.transmat_prior is None:
                transmat_init = uniform_prior
            else:
                transmat_init = self.transmat_prior
            self.transmat_prior_ = np.full(
                (nc, nc), transmat_init)
            self.transmat_posterior_ = random_state.dirichlet(
                np.full(nc, uniform_prior), size=nc)
            self.transmat_posterior_ *= sum(lengths) / nc

        n_fit_scalars_per_param = self._get_n_fit_scalars_per_param()
        if n_fit_scalars_per_param is not None:
            n_fit_scalars = sum(
                n_fit_scalars_per_param[p] for p in self.params)
            if X.size < n_fit_scalars:
                _log.warning(
                    "Fitting a model with %d free scalar parameters with only "
                    "%d data points will result in a degenerate solution.",
                    n_fit_scalars, X.size)

    # For Variational Inference, we compute the forward/backward algorithm
    # using subnormalized probabilities.
    def _fit_scaling(self, X):
        frameprob = self._compute_subnorm_likelihood(X)
        logprob, fwdlattice, scaling_factors = _hmmc.forward_scaling(
            self.startprob_subnorm_, self.transmat_subnorm_, frameprob)

        bwdlattice = _hmmc.backward_scaling(
            self.startprob_subnorm_, self.transmat_subnorm_,
            frameprob, scaling_factors)
        posteriors = self._compute_posteriors_scaling(fwdlattice, bwdlattice)
        return frameprob, logprob, posteriors, fwdlattice, bwdlattice

    def _fit_log(self, X):
        framelogprob = self._compute_subnorm_log_likelihood(X)
        logprob, fwdlattice = _hmmc.forward_log(
            self.startprob_subnorm_, self.transmat_subnorm_, framelogprob)
        bwdlattice = _hmmc.backward_log(
            self.startprob_subnorm_, self.transmat_subnorm_, framelogprob)
        posteriors = self._compute_posteriors_log(fwdlattice, bwdlattice)
        return framelogprob, logprob, posteriors, fwdlattice, bwdlattice

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

        self.startprob_prior_ = np.asarray(self.startprob_prior_)
        if len(self.startprob_prior_) != nc:
            raise ValueError("startprob_prior_ must have length n_components")
        self.startprob_posterior_ = np.asarray(self.startprob_posterior_)
        if len(self.startprob_posterior_) != nc:
            raise ValueError("startprob_posterior_ must have length "
                             "n_components")

        self.transmat_prior_ = np.asarray(self.transmat_prior_)
        if self.transmat_prior_.shape != (nc, nc):
            raise ValueError("transmat_prior_ must have shape "
                             "(n_components, n_components)")
        self.transmat_posterior_ = np.asarray(self.transmat_posterior_)
        if self.transmat_posterior_.shape != (nc, nc):
            raise ValueError("transmat_posterior_ must have shape "
                             "(n_components, n_components)")

    def _compute_subnorm_likelihood(self, X):
        if (self._compute_subnorm_log_likelihood !=  # prevent recursion
                __class__._compute_subnorm_log_likelihood.__get__(self)):
            return np.exp(self._compute_subnorm_log_likelihood(X))
        else:
            raise NotImplementedError("Must be overridden in subclass")

    def _compute_subnorm_log_likelihood(self, X):
        if (self._compute_subnorm_likelihood !=  # prevent recursion
                __class__._compute_subnorm_likelihood.__get__(self)):
            return np.log(self._compute_subnorm_likelihood(X))
        else:
            raise NotImplementedError("Must be overridden in subclass")

    def _accumulate_sufficient_statistics_scaling(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
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

    def _accumulate_sufficient_statistics_log(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
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
                fwdlattice, self.transmat_subnorm_, bwdlattice,
                lattice)
            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)

    def _estep_begin(self):
        """
        Update the subnormalized model parameters.  Called at the beginning of
        each iteration of fit()
        """
        startprob_log_subnorm = (
            special.digamma(self.startprob_posterior_)
            - special.digamma(self.startprob_posterior_.sum()))
        self.startprob_subnorm_ = np.exp(startprob_log_subnorm)

        transmat_log_subnorm = (
            special.digamma(self.transmat_posterior_)
            - special.digamma(self.transmat_posterior_.sum(axis=1)[:, None]))
        self.transmat_subnorm_ = np.exp(transmat_log_subnorm)

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
            # For compatability in _AbstractHMM
            self.startprob_ = (self.startprob_posterior_
                               / self.startprob_posterior_.sum())
        if 't' in self.params:
            self.transmat_posterior_ = self.transmat_prior_ + stats['trans']
            # For compatability in _AbstractHMM
            self.transmat_ = (self.transmat_posterior_
                              / self.transmat_posterior_.sum(axis=1)[:, None])

    def _compute_lower_bound(self, curr_logprob):
        """
        Compute the Variational Lower Bound of the model as currently
        configured.

        Following the pattern elsewhere, derived implementations should call
        this method to get the contribution of the current log_prob,
        transmat, and startprob towards the lower bound

        Parameters
        ----------
        curr_logprob : float
                       The current log probability of the data as computed at
                       the subnormalized model parameters.

        Returns
        -------
        lower_bound: float
                     Returns the computed lower bound contribution of the
                     log_prob, startprob, and transmat.
        """
        # Get the contribution from the state transitions,
        # initial probabilities, and the likelihood of the sequences
        startprob_lower_bound = -_kl.kl_dirichlet(
            self.startprob_posterior_, self.startprob_prior_)
        transmat_lower_bound = 0
        for i in range(self.n_components):
            transmat_lower_bound -= _kl.kl_dirichlet(
                self.transmat_posterior_[i], self.transmat_prior_[i])
        return startprob_lower_bound + transmat_lower_bound + curr_logprob
