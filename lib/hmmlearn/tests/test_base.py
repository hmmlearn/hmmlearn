import numpy as np
import pytest
from scipy import special

from hmmlearn.base import _BaseHMM, ConvergenceMonitor


class TestMonitor:
    def test_converged_by_iterations(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=2, verbose=False)
        assert not m.converged
        m.report(-0.01)
        assert not m.converged
        m.report(-0.1)
        assert m.converged

    def test_converged_by_logprob(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=False)
        for logprob in [-0.03, -0.02, -0.01]:
            m.report(logprob)
            assert not m.converged

        m.report(-0.0101)
        assert m.converged

    def test_reset(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=False)
        m.iter = 1
        m.history.append(-0.01)
        m._reset()
        assert m.iter == 0
        assert not m.history

    def test_report_first_iteration(self, capsys):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=True)
        m.report(-0.01)
        out, err = capsys.readouterr()
        assert not out
        expected = m._template.format(iter=1, logprob=-0.01, delta=np.nan)
        assert err.splitlines() == [expected]

    def test_report(self, capsys):
        n_iter = 10
        m = ConvergenceMonitor(tol=1e-3, n_iter=n_iter, verbose=True)
        for i in reversed(range(n_iter)):
            m.report(-0.01 * i)

        out, err = capsys.readouterr()
        assert not out
        assert len(err.splitlines()) == n_iter
        assert len(m.history) == n_iter


class StubHMM(_BaseHMM):
    """An HMM with hardcoded observation probabilities."""
    def _compute_log_likelihood(self, X):
        return self.framelogprob


class TestBaseAgainstWikipedia:
    def setup_method(self, method):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        self.framelogprob = np.log([[0.9, 0.2],
                                    [0.9, 0.2],
                                    [0.1, 0.8],
                                    [0.9, 0.2],
                                    [0.9, 0.2]])

        h = StubHMM(2)
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.startprob_ = [0.5, 0.5]
        h.framelogprob = self.framelogprob
        self.hmm = h

    def test_do_forward_pass(self):
        logprob, fwdlattice = self.hmm._do_forward_pass(self.framelogprob)

        reflogprob = -3.3725
        assert round(logprob, 4) == reflogprob
        reffwdlattice = np.array([[0.4500, 0.1000],
                                  [0.3105, 0.0410],
                                  [0.0230, 0.0975],
                                  [0.0408, 0.0150],
                                  [0.0298, 0.0046]])
        assert np.allclose(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_backward_pass(self):
        bwdlattice = self.hmm._do_backward_pass(self.framelogprob)

        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        assert np.allclose(np.exp(bwdlattice), refbwdlattice, 4)

    def test_do_viterbi_pass(self):
        logprob, state_sequence = self.hmm._do_viterbi_pass(self.framelogprob)

        refstate_sequence = [0, 0, 1, 0, 0]
        assert np.allclose(state_sequence, refstate_sequence)

        reflogprob = -4.4590
        assert round(logprob, 4) == reflogprob

    def test_score_samples(self):
        # ``StubHMM` ignores the values in ```X``, so we just pass in an
        # array of the appropriate shape.
        logprob, posteriors = self.hmm.score_samples(self.framelogprob)
        assert np.allclose(posteriors.sum(axis=1), np.ones(len(posteriors)))

        reflogprob = -3.3725
        assert round(logprob, 4) == reflogprob

        refposteriors = np.array([[0.8673, 0.1327],
                                  [0.8204, 0.1796],
                                  [0.3075, 0.6925],
                                  [0.8204, 0.1796],
                                  [0.8673, 0.1327]])
        assert np.allclose(posteriors, refposteriors, atol=1e-4)


class TestBaseConsistentWithGMM:
    def setup_method(self, method):
        n_components = 8
        n_samples = 10

        self.framelogprob = np.log(np.random.random((n_samples, n_components)))

        h = StubHMM(n_components)
        h.framelogprob = self.framelogprob

        # If startprob and transmat are uniform across all states (the
        # default), the transitions are uninformative - the model
        # reduces to a GMM with uniform mixing weights (in terms of
        # posteriors, not likelihoods).
        h.startprob_ = np.ones(n_components) / n_components
        h.transmat_ = np.ones((n_components, n_components)) / n_components

        self.hmm = h

    def test_score_samples(self):
        logprob, hmmposteriors = self.hmm.score_samples(self.framelogprob)

        n_samples, n_components = self.framelogprob.shape
        assert np.allclose(hmmposteriors.sum(axis=1), np.ones(n_samples))

        norm = special.logsumexp(self.framelogprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.framelogprob
                               - np.tile(norm, (1, n_components)))
        assert np.allclose(hmmposteriors, gmmposteriors)

    def test_decode(self):
        _logprob, state_sequence = self.hmm.decode(self.framelogprob)

        n_samples, n_components = self.framelogprob.shape
        norm = special.logsumexp(self.framelogprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.framelogprob -
                               np.tile(norm, (1, n_components)))
        gmmstate_sequence = gmmposteriors.argmax(axis=1)
        assert np.allclose(state_sequence, gmmstate_sequence)


def test_base_hmm_attributes():
    n_components = 20
    startprob = np.random.random(n_components)
    startprob /= startprob.sum()
    transmat = np.random.random((n_components, n_components))
    transmat /= np.tile(transmat.sum(axis=1)[:, np.newaxis], (1, n_components))

    h = StubHMM(n_components)

    assert h.n_components == n_components

    h.startprob_ = startprob
    assert np.allclose(h.startprob_, startprob)

    with pytest.raises(ValueError):
        h.startprob_ = 2 * startprob
        h._check()
    with pytest.raises(ValueError):
        h.startprob_ = []
        h._check()
    with pytest.raises(ValueError):
        h.startprob_ = np.zeros((n_components - 2, 2))
        h._check()

    h.startprob_ = startprob
    h.transmat_ = transmat
    assert np.allclose(h.transmat_, transmat)
    with pytest.raises(ValueError):
        h.transmat_ = 2 * transmat
        h._check()
    with pytest.raises(ValueError):
        h.transmat_ = []
        h._check()
    with pytest.raises(ValueError):
        h.transmat_ = np.zeros((n_components - 2, n_components))
        h._check()


def test_stationary_distribution():
    n_components = 10
    h = StubHMM(n_components)
    transmat = np.random.random((n_components, n_components))
    transmat /= np.tile(transmat.sum(axis=1)[:, np.newaxis], (1, n_components))
    h.transmat_ = transmat
    stationary = h.get_stationary_distribution()
    assert stationary.dtype == float
    assert np.dot(h.get_stationary_distribution().T, h.transmat_) \
        == pytest.approx(stationary)
