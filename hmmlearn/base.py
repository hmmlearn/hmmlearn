from __future__ import print_function

import string
import sys
from collections import deque

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.extmath import logsumexp

from . import _hmmc
from .utils import normalize


decoder_algorithms = frozenset(("viterbi", "map"))


ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf


class ConvergenceMonitor(object):
    """Monitors and reports convergence to :data:`sys.stderr`.

    Parameters
    ----------
    thresh : double
        Convergence threshold. The algorithm has convereged eitehr if
        the maximum number of iterations is reached or the log probability
        improvement between the two consecutive iterations is less than
        threshold.

    n_iter : int
        Maximum number of iterations to perform.

    verbose : bool
        If ``True`` then per-iteration convergence reports are printed,
        otherwise the monitor is mute.

    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.

    iter : int
        Number of iterations performed while training the model.
    """
    fmt = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, thresh, n_iter, verbose):
        self.thresh = thresh
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 1

    def report(self, logprob):
        if self.history and self.verbose:
            delta = logprob - self.history[-1]
            message = self.fmt.format(
                iter=self.iter, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        return (self.iter == self.n_iter or
                (len(self.history) == 2 and
                 self.history[1] - self.history[0] < self.thresh))


class _BaseHMM(BaseEstimator):
    """Hidden Markov Model base class.

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Attributes
    ----------
    n_components : int
        Number of states in the model.

    transmat : array, shape (`n_components`, `n_components`)
        Matrix of transition probabilities between states.

    startprob : array, shape ('n_components`,)
        Initial state occupation distribution.

    transmat_prior : array, shape (`n_components`, `n_components`)
        Matrix of prior transition probabilities between states.

    startprob_prior : array, shape ('n_components`,)
        Initial state occupation prior distribution.

    algorithm : string, one of the decoder_algorithms
        Decoder algorithm.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Maximum number of iterations to perform.

    thresh : float, optional
        Convergence threshold.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, and other characters for subclass-specific
        emmission parameters. Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emmission parameters. Defaults to all
        parameters.

    See Also
    --------
    GMM : Gaussian mixture model
    """

    # This class implements the public interface to all HMMs that
    # derive from it, including all of the machinery for the
    # forward-backward and Viterbi algorithms.  Subclasses need only
    # implement _generate_sample_from_state(), _compute_log_likelihood(),
    # _init(), _initialize_sufficient_statistics(),
    # _accumulate_sufficient_statistics(), and _do_mstep(), all of
    # which depend on the specific emission distribution.
    #
    # Subclasses will probably also want to implement properties for
    # the emission distribution parameters to expose them publicly.

    def __init__(self, n_components=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        # TODO: move all validation from descriptors to 'fit' and 'predict'.
        self.n_components = n_components
        self.n_iter = n_iter
        self.thresh = thresh
        self.monitor_ = ConvergenceMonitor(thresh, n_iter, verbose)
        self.params = params
        self.init_params = init_params
        self.startprob_ = startprob
        self.startprob_prior = startprob_prior
        self.transmat_ = transmat
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state

        if transmat is not None:
            self.transmat_mask = (transmat > np.finfo(float).eps)
        if startprob is not None:
            self.startprob_mask = (startprob > np.finfo(float).eps)

    def eval(self, X):
        return self.score_samples(X)

    def score_samples(self, obs):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        logprob : float
            Log likelihood of the sequence ``obs``.

        posteriors : array_like, shape (n, n_components)
            Posterior probabilities of each state for each
            observation

        See Also
        --------
        score : Compute the log probability under the model
        decode : Find most likely state sequence corresponding to a `obs`
        """
        obs = np.asarray(obs)
        framelogprob = self._compute_log_likelihood(obs)
        logprob, fwdlattice = self._do_forward_pass(framelogprob)
        bwdlattice = self._do_backward_pass(framelogprob)
        gamma = fwdlattice + bwdlattice
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
        posteriors += np.finfo(np.float64).eps
        posteriors /= np.sum(posteriors, axis=1).reshape((-1, 1))
        return logprob, posteriors

    def score(self, obs):
        """Compute the log probability under the model.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : float
            Log likelihood of the ``obs``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors

        decode : Find most likely state sequence corresponding to a `obs`
        """
        obs = np.asarray(obs)
        framelogprob = self._compute_log_likelihood(obs)
        logprob, _ = self._do_forward_pass(framelogprob)
        return logprob

    def _decode_viterbi(self, obs):
        """Find most likely state sequence corresponding to ``obs``.

        Uses the Viterbi algorithm.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        viterbi_logprob : float
            Log probability of the maximum likelihood path through the HMM.

        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model
        """
        obs = np.asarray(obs)
        framelogprob = self._compute_log_likelihood(obs)
        viterbi_logprob, state_sequence = self._do_viterbi_pass(framelogprob)
        return viterbi_logprob, state_sequence

    def _decode_map(self, obs):
        """Find most likely state sequence corresponding to `obs`.

        Uses the maximum a posteriori estimation.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        map_logprob : float
            Log probability of the maximum likelihood path through the HMM
        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        _, posteriors = self.score_samples(obs)
        state_sequence = np.argmax(posteriors, axis=1)
        map_logprob = np.max(posteriors, axis=1).sum()
        return map_logprob, state_sequence

    def decode(self, obs, algorithm="viterbi"):
        """Find most likely state sequence corresponding to ``obs``.
        Uses the selected algorithm for decoding.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        algorithm : string, one of the `decoder_algorithms`
            decoder algorithm to be used

        Returns
        -------
        logprob : float
            Log probability of the maximum likelihood path through the HMM

        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model.
        """
        if self.algorithm in decoder_algorithms:
            algorithm = self.algorithm
        elif algorithm in decoder_algorithms:
            algorithm = algorithm
        decoder = {"viterbi": self._decode_viterbi,
                   "map": self._decode_map}
        logprob, state_sequence = decoder[algorithm](obs)
        return logprob, state_sequence

    def predict(self, obs, algorithm="viterbi"):
        """Find most likely state sequence corresponding to `obs`.

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        state_sequence : array_like, shape (n,)
            Index of the most likely states for each observation
        """
        _, state_sequence = self.decode(obs, algorithm)
        return state_sequence

    def predict_proba(self, obs):
        """Compute the posterior probability for each state in the model

        Parameters
        ----------
        obs : array_like, shape (n, n_features)
            Sequence of n_features-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        T : array-like, shape (n, n_components)
            Returns the probability of the sample for each state in the model.
        """
        _, posteriors = self.score_samples(obs)
        return posteriors

    def sample(self, n=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        random_state: RandomState or an int seed (0 by default)
            A random number generator instance. If None is given, the
            object's random_state is used

        Returns
        -------
        (obs, hidden_states)
        obs : array_like, length `n` List of samples
        hidden_states : array_like, length `n` List of hidden states
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_pdf = self.startprob_
        startprob_cdf = np.cumsum(startprob_pdf)
        transmat_pdf = self.transmat_
        transmat_cdf = np.cumsum(transmat_pdf, 1)

        # Initial state.
        rand = random_state.rand()
        currstate = (startprob_cdf > rand).argmax()
        hidden_states = [currstate]
        obs = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for _ in range(n - 1):
            rand = random_state.rand()
            currstate = (transmat_cdf[currstate] > rand).argmax()
            hidden_states.append(currstate)
            obs.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.array(obs), np.array(hidden_states, dtype=int)

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
        """
        self._init(obs, self.init_params)

        for i in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for seq in obs:
                framelogprob = self._compute_log_likelihood(seq)
                lpr, fwdlattice = self._do_forward_pass(framelogprob)
                bwdlattice = self._do_backward_pass(framelogprob)
                gamma = fwdlattice + bwdlattice
                posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
                curr_logprob += lpr
                self._accumulate_sufficient_statistics(
                    stats, seq, framelogprob, posteriors, fwdlattice,
                    bwdlattice, self.params)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

            self._do_mstep(stats, self.params)

        return self

    def _get_algorithm(self):
        "decoder algorithm"
        return self._algorithm

    def _set_algorithm(self, algorithm):
        if algorithm not in decoder_algorithms:
            raise ValueError("algorithm must be one of the decoder_algorithms")
        self._algorithm = algorithm

    algorithm = property(_get_algorithm, _set_algorithm)

    def _get_startprob(self):
        """Mixing startprob for each state."""
        return np.exp(self._log_startprob)

    def _set_startprob(self, startprob):
        if startprob is None:
            startprob = np.tile(1.0 / self.n_components, self.n_components)
        else:
            startprob = np.asarray(startprob, dtype=np.float)

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(startprob):
            normalize(startprob)

        if len(startprob) != self.n_components:
            raise ValueError('startprob must have length n_components')
        if not np.allclose(np.sum(startprob), 1.0):
            raise ValueError('startprob must sum to 1.0')

        self._log_startprob = np.log(np.asarray(startprob).copy())

    startprob_ = property(_get_startprob, _set_startprob)

    def _get_transmat(self):
        """Matrix of transition probabilities."""
        return np.exp(self._log_transmat)

    def _set_transmat(self, transmat):
        if transmat is None:
            transmat = np.tile(1.0 / self.n_components,
                               (self.n_components, self.n_components))

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(transmat):
            normalize(transmat, axis=1)

        if (np.asarray(transmat).shape
                != (self.n_components, self.n_components)):
            raise ValueError('transmat must have shape '
                             '(n_components, n_components)')
        if not np.all(np.allclose(np.sum(transmat, axis=1), 1.0)):
            raise ValueError('Rows of transmat must sum to 1.0')

        self._log_transmat = np.log(np.asarray(transmat).copy())
        underflow_idx = np.isnan(self._log_transmat)
        self._log_transmat[underflow_idx] = NEGINF

    transmat_ = property(_get_transmat, _set_transmat)

    def _do_viterbi_pass(self, framelogprob):
        n_observations, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc._viterbi(
            n_observations, n_components, self._log_startprob,
            self._log_transmat, framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):
        n_observations, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_observations, n_components))
        _hmmc._forward(n_observations, n_components, self._log_startprob,
                       self._log_transmat, framelogprob, fwdlattice)
        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        n_observations, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_observations, n_components))
        _hmmc._backward(n_observations, n_components, self._log_startprob,
                        self._log_transmat, framelogprob, bwdlattice)
        return bwdlattice

    def _compute_log_likelihood(self, obs):
        pass

    def _generate_sample_from_state(self, state, random_state=None):
        pass

    def _init(self, obs, params):
        if 's' in params:
            self.startprob_.fill(1.0 / self.n_components)
        if 't' in params:
            self.transmat_.fill(1.0 / self.n_components)

    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components))}
        return stats

    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        stats['nobs'] += 1
        if 's' in params:
            stats['start'] += posteriors[0]
        if 't' in params:
            n_observations, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_observations <= 1:
                return

            lneta = np.zeros((n_observations - 1, n_components, n_components))
            _hmmc._compute_lneta(n_observations, n_components, fwdlattice,
                                 self._log_transmat, bwdlattice, framelogprob,
                                 lneta)
            stats['trans'] += np.exp(logsumexp(lneta, axis=0))

    def _do_mstep(self, stats, params):
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        if self.startprob_prior is None:
            self.startprob_prior = 1.0
        if self.transmat_prior is None:
            self.transmat_prior = 1.0

        if 's' in params:
            self.startprob_ = normalize(self.startprob_prior - 1.0 + stats['start'], mask=self.startprob_mask)
        if 't' in params:
            transmat_ = normalize(self.transmat_prior - 1.0 + stats['trans'], axis=1, mask=self.transmat_mask)
            self.transmat_ = transmat_
