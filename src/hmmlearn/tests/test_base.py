import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import special

from hmmlearn.base import BaseHMM, ConvergenceMonitor
from hmmlearn import _hmmc


class TestMonitor:
    def test_converged_by_iterations(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=2, verbose=False)
        assert not m.converged
        m.report(-0.01)
        assert not m.converged
        m.report(-0.1)
        assert m.converged

    def test_converged_by_log_prob(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=False)
        for log_prob in [-0.03, -0.02, -0.01]:
            m.report(log_prob)
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
        expected = m._template.format(iter=1, log_prob=-0.01, delta=np.nan)
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


class StubHMM(BaseHMM):
    """An HMM with hardcoded observation probabilities."""
    def _compute_log_likelihood(self, X):
        return self.log_frameprob


class TestBaseAgainstWikipedia:
    def setup_method(self, method):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        self.frameprob = np.asarray([[0.9, 0.2],
                                     [0.9, 0.2],
                                     [0.1, 0.8],
                                     [0.9, 0.2],
                                     [0.9, 0.2]])
        self.log_frameprob = np.log(self.frameprob)
        h = StubHMM(2)
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.startprob_ = [0.5, 0.5]
        h.log_frameprob = self.log_frameprob
        h.frameprob = self.frameprob
        self.hmm = h

    def test_do_forward_scaling_pass(self):
        log_prob, fwdlattice, scaling_factors = _hmmc.forward_scaling(
            self.hmm.startprob_, self.hmm.transmat_, self.frameprob)
        ref_log_prob = -3.3725
        assert round(log_prob, 4) == ref_log_prob
        reffwdlattice = np.exp([[0.4500, 0.1000],
                                [0.3105, 0.0410],
                                [0.0230, 0.0975],
                                [0.0408, 0.0150],
                                [0.0298, 0.0046]])
        assert_allclose(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_forward_pass(self):
        log_prob, fwdlattice = _hmmc.forward_log(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob)

        ref_log_prob = -3.3725
        assert round(log_prob, 4) == ref_log_prob
        reffwdlattice = np.array([[0.4500, 0.1000],
                                  [0.3105, 0.0410],
                                  [0.0230, 0.0975],
                                  [0.0408, 0.0150],
                                  [0.0298, 0.0046]])
        assert_allclose(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_backward_scaling_pass(self):
        log_prob, fwdlattice, scaling_factors = _hmmc.forward_scaling(
            self.hmm.startprob_, self.hmm.transmat_, self.frameprob)
        bwdlattice = _hmmc.backward_scaling(self.hmm.startprob_,
            self.hmm.transmat_, self.frameprob, scaling_factors)
        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        scaling_factors = np.cumprod(scaling_factors[::-1])[::-1]
        bwdlattice_scaled = bwdlattice / scaling_factors[:, None]
        # Answer will be equivalent when the scaling factor is accounted for
        assert_allclose(bwdlattice_scaled, refbwdlattice, 4)

    def test_do_backward_log_pass(self):
        bwdlattice = _hmmc.backward_log(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob)
        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        assert_allclose(np.exp(bwdlattice), refbwdlattice, 4)

    def test_do_viterbi_pass(self):
        log_prob, state_sequence = _hmmc.viterbi(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob)
        refstate_sequence = [0, 0, 1, 0, 0]
        assert_allclose(state_sequence, refstate_sequence)
        ref_log_prob = -4.4590
        assert round(log_prob, 4) == ref_log_prob

    def test_score_samples(self):
        # ``StubHMM` ignores the values in ```X``, so we just pass in an
        # array of the appropriate shape.
        log_prob, posteriors = self.hmm.score_samples(self.log_frameprob)
        assert_allclose(posteriors.sum(axis=1), np.ones(len(posteriors)))
        ref_log_prob = -3.3725
        assert round(log_prob, 4) == ref_log_prob
        refposteriors = np.array([[0.8673, 0.1327],
                                  [0.8204, 0.1796],
                                  [0.3075, 0.6925],
                                  [0.8204, 0.1796],
                                  [0.8673, 0.1327]])
        assert_allclose(posteriors, refposteriors, atol=1e-4)

    def test_generate_samples(self):
        X0, Z0 = self.hmm.sample(n_samples=10)
        X, Z = self.hmm.sample(n_samples=10, currstate=Z0[-1])
        assert len(Z0) == len(Z) == 10 and Z[0] == Z0[-1]


class TestBaseConsistentWithGMM:
    def setup_method(self, method):
        n_components = 8
        n_samples = 10

        self.log_frameprob = np.log(
            np.random.random((n_samples, n_components)))

        h = StubHMM(n_components)
        h.log_frameprob = self.log_frameprob

        # If startprob and transmat are uniform across all states (the
        # default), the transitions are uninformative - the model
        # reduces to a GMM with uniform mixing weights (in terms of
        # posteriors, not likelihoods).
        h.startprob_ = np.ones(n_components) / n_components
        h.transmat_ = np.ones((n_components, n_components)) / n_components

        self.hmm = h

    def test_score_samples(self):
        log_prob, hmmposteriors = self.hmm.score_samples(self.log_frameprob)

        n_samples, n_components = self.log_frameprob.shape
        assert_allclose(hmmposteriors.sum(axis=1), np.ones(n_samples))

        norm = special.logsumexp(self.log_frameprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.log_frameprob
                               - np.tile(norm, (1, n_components)))
        assert_allclose(hmmposteriors, gmmposteriors)

    def test_decode(self):
        _log_prob, state_sequence = self.hmm.decode(self.log_frameprob)

        n_samples, n_components = self.log_frameprob.shape
        norm = special.logsumexp(self.log_frameprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.log_frameprob
                               - np.tile(norm, (1, n_components)))
        gmmstate_sequence = gmmposteriors.argmax(axis=1)
        assert_allclose(state_sequence, gmmstate_sequence)


def test_base_hmm_attributes():
    n_components = 20
    startprob = np.random.random(n_components)
    startprob /= startprob.sum()
    transmat = np.random.random((n_components, n_components))
    transmat /= np.tile(transmat.sum(axis=1)[:, np.newaxis], (1, n_components))

    h = StubHMM(n_components)

    assert h.n_components == n_components

    h.startprob_ = startprob
    assert_allclose(h.startprob_, startprob)

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
    assert_allclose(h.transmat_, transmat)
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
    assert (h.get_stationary_distribution().T @ h.transmat_
            == pytest.approx(stationary))
