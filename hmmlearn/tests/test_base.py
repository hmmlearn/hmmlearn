from __future__ import print_function

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils.extmath import logsumexp

from hmmlearn import hmm

np.seterr(all='warn')


class StubHMM(hmm._BaseHMM):
    def _compute_log_likelihood(self, X):
        return self.framelogprob

    def _generate_sample_from_state(self):
        pass

    def _init(self):
        pass


class TestBaseHMM(TestCase):
    def setUp(self):
        self.prng = np.random.RandomState(10)

    def setup_example_hmm(self):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        h = StubHMM(2)
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.startprob_ = [0.5, 0.5]
        framelogprob = np.log([[0.9, 0.2],
                               [0.9, 0.2],
                               [0.1, 0.8],
                               [0.9, 0.2],
                               [0.9, 0.2]])
        # Add dummy observations to stub.
        h.framelogprob = framelogprob
        return h, framelogprob

    def test_set_startprob(self):
        h, framelogprob = self.setup_example_hmm()
        startprob = np.array([0.0, 1.0])
        h.startprob_ = startprob
        assert np.allclose(startprob, h.startprob_)

    def test_set_transmat(self):
        h, framelogprob = self.setup_example_hmm()
        transmat = np.array([[0.8, 0.2], [0.0, 1.0]])
        h.transmat_ = transmat
        assert np.allclose(transmat, h.transmat_)

    def test_do_forward_pass(self):
        h, framelogprob = self.setup_example_hmm()

        logprob, fwdlattice = h._do_forward_pass(framelogprob)

        reflogprob = -3.3725
        self.assertAlmostEqual(logprob, reflogprob, places=4)

        reffwdlattice = np.array([[0.4500, 0.1000],
                                  [0.3105, 0.0410],
                                  [0.0230, 0.0975],
                                  [0.0408, 0.0150],
                                  [0.0298, 0.0046]])
        assert_array_almost_equal(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_backward_pass(self):
        h, framelogprob = self.setup_example_hmm()

        bwdlattice = h._do_backward_pass(framelogprob)

        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        assert_array_almost_equal(np.exp(bwdlattice), refbwdlattice, 4)

    def test_do_viterbi_pass(self):
        h, framelogprob = self.setup_example_hmm()

        logprob, state_sequence = h._do_viterbi_pass(framelogprob)

        refstate_sequence = [0, 0, 1, 0, 0]
        assert_array_equal(state_sequence, refstate_sequence)

        reflogprob = -4.4590
        self.assertAlmostEqual(logprob, reflogprob, places=4)

    def test_score_samples(self):
        h, framelogprob = self.setup_example_hmm()
        nobs = len(framelogprob)

        logprob, posteriors = h.score_samples(framelogprob)

        assert_array_almost_equal(posteriors.sum(axis=1), np.ones(nobs))

        reflogprob = -3.3725
        self.assertAlmostEqual(logprob, reflogprob, places=4)

        refposteriors = np.array([[0.8673, 0.1327],
                                  [0.8204, 0.1796],
                                  [0.3075, 0.6925],
                                  [0.8204, 0.1796],
                                  [0.8673, 0.1327]])
        assert_array_almost_equal(posteriors, refposteriors, decimal=4)

    def test_hmm_score_samples_consistent_with_gmm(self):
        n_components = 8
        nobs = 10
        h = StubHMM(n_components)

        # Add dummy observations to stub.
        framelogprob = np.log(self.prng.rand(nobs, n_components))
        h.framelogprob = framelogprob

        # If startprob and transmat are uniform across all states (the
        # default), the transitions are uninformative - the model
        # reduces to a GMM with uniform mixing weights (in terms of
        # posteriors, not likelihoods).
        h.startprob_ = np.ones(n_components) / n_components
        h.transmat_ = np.ones((n_components, n_components)) / n_components
        logprob, hmmposteriors = h.score_samples(framelogprob)

        assert_array_almost_equal(hmmposteriors.sum(axis=1), np.ones(nobs))

        norm = logsumexp(framelogprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(framelogprob - np.tile(norm, (1, n_components)))
        assert_array_almost_equal(hmmposteriors, gmmposteriors)

    def test_hmm_decode_consistent_with_gmm(self):
        n_components = 8
        nobs = 10
        h = StubHMM(n_components)

        # Add dummy observations to stub.
        framelogprob = np.log(self.prng.rand(nobs, n_components))
        h.framelogprob = framelogprob

        # If startprob and transmat are uniform across all states (the
        # default), the transitions are uninformative - the model
        # reduces to a GMM with uniform mixing weights (in terms of
        # posteriors, not likelihoods).
        h.startprob_ = np.ones(n_components) / n_components
        h.transmat_ = np.ones((n_components, n_components)) / n_components
        viterbi_ll, state_sequence = h.decode(framelogprob)

        norm = logsumexp(framelogprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(framelogprob - np.tile(norm, (1, n_components)))
        gmmstate_sequence = gmmposteriors.argmax(axis=1)
        assert_array_equal(state_sequence, gmmstate_sequence)

    def test_base_hmm_attributes(self):
        n_components = 20
        startprob = self.prng.rand(n_components)
        startprob = startprob / startprob.sum()
        transmat = self.prng.rand(n_components, n_components)
        transmat /= np.tile(transmat.sum(axis=1)
                            [:, np.newaxis], (1, n_components))

        h = StubHMM(n_components)

        self.assertEqual(h.n_components, n_components)

        h.startprob_ = startprob
        assert_array_almost_equal(h.startprob_, startprob)

        with self.assertRaises(ValueError):
            h.startprob_ = 2 * startprob
            h._check()
        with self.assertRaises(ValueError):
            h.startprob_ = []
            h._check()
        with self.assertRaises(ValueError):
            h.startprob_ = np.zeros((n_components - 2, 2))
            h._check()

        h.startprob_ = startprob
        h.transmat_ = transmat
        assert_array_almost_equal(h.transmat_, transmat)
        with self.assertRaises(ValueError):
            h.transmat_ = 2 * transmat
            h._check()
        with self.assertRaises(ValueError):
            h.transmat_ = []
            h._check()
        with self.assertRaises(ValueError):
            h.transmat_ = np.zeros((n_components - 2, n_components))
            h._check()
