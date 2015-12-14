from __future__ import print_function

from unittest import TestCase

import numpy as np
import pytest
from sklearn.datasets.samples_generator import make_spd_matrix
from sklearn.mixture import GMM
from sklearn.utils import check_random_state

from hmmlearn import hmm
from hmmlearn.utils import normalize

np.seterr(all='warn')


def fit_hmm_and_monitor_log_likelihood(h, X, lengths=None, n_iter=1):
    h.n_iter = 1        # make sure we do a single iteration at a time
    h.init_params = ''  # and don't re-init params
    loglikelihoods = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        h.fit(X, lengths=lengths)
        loglikelihoods[i] = h.score(X, lengths=lengths)
    return loglikelihoods


class GaussianHMMTestMixin(object):
    covariance_type = None  # set by subclasses

    def setUp(self):
        self.prng = prng = np.random.RandomState(10)
        self.n_components = n_components = 3
        self.n_features = n_features = 3
        self.startprob = prng.rand(n_components)
        self.startprob = self.startprob / self.startprob.sum()
        self.transmat = prng.rand(n_components, n_components)
        self.transmat /= np.tile(self.transmat.sum(axis=1)[:, np.newaxis],
                                 (1, n_components))
        self.means = prng.randint(-20, 20, (n_components, n_features))
        self.covars = {
            'spherical': (1.0 + 2 * np.dot(prng.rand(n_components, 1),
                                           np.ones((1, n_features)))) ** 2,
            'tied': (make_spd_matrix(n_features, random_state=0)
                     + np.eye(n_features)),
            'diag': (1.0 + 2 * prng.rand(n_components, n_features)) ** 2,
            'full': np.array([make_spd_matrix(n_features, random_state=0)
                              + np.eye(n_features)
                              for x in range(n_components)]),
        }
        self.expanded_covars = {
            'spherical': [np.eye(n_features) * cov
                          for cov in self.covars['spherical']],
            'diag': [np.diag(cov) for cov in self.covars['diag']],
            'tied': [self.covars['tied']] * n_components,
            'full': self.covars['full'],
        }

    def test_bad_covariance_type(self):
        with pytest.raises(ValueError):
            h = hmm.GaussianHMM(20, covariance_type='badcovariance_type')
            h.means_ = self.means
            h.covars_ = []
            h.startprob_ = self.startprob
            h.transmat_ = self.transmat
            h._check()

    def test_score_samples_and_decode(self):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            init_params="st")
        h.means_ = self.means
        h.covars_ = self.covars[self.covariance_type]

        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        h.means_ = 20 * h.means_

        gaussidx = np.repeat(np.arange(self.n_components), 5)
        n_samples = len(gaussidx)
        X = self.prng.randn(n_samples, self.n_features) + h.means_[gaussidx]
        h._init(X)
        ll, posteriors = h.score_samples(X)

        self.assertEqual(posteriors.shape, (n_samples, self.n_components))
        assert np.allclose(posteriors.sum(axis=1), np.ones(n_samples))

        viterbi_ll, stateseq = h.decode(X)
        assert np.allclose(stateseq, gaussidx)

    def test_sample(self, n=1000):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        h.means_ = 20 * self.means
        h.covars_ = np.maximum(self.covars[self.covariance_type], 0.1)

        X, state_sequence = h.sample(n, random_state=self.prng)
        self.assertEqual(X.shape, (n, self.n_features))
        self.assertEqual(len(state_sequence), n)

    def test_fit(self, params='stmc', n_iter=5, **kwargs):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        h.startprob_ = self.startprob
        h.transmat_ = normalize(
            self.transmat + np.diag(self.prng.rand(self.n_components)), 1)
        h.means_ = 20 * self.means
        h.covars_ = self.covars[self.covariance_type]

        lengths = [10] * 10
        X, _state_sequence = h.sample(sum(lengths), random_state=self.prng)

        # Mess up the parameters and see if we can re-learn them.
        h.n_iter = 0
        h.fit(X, lengths=lengths)

        trainll = fit_hmm_and_monitor_log_likelihood(
            h, X, lengths=lengths, n_iter=n_iter)

        # Check that the log-likelihood is always increasing during training.
        diff = np.diff(trainll)
        message = ("Decreasing log-likelihood for {0} covariance: {1}"
                   .format(self.covariance_type, diff))
        self.assertTrue(np.all(diff >= -1e-6), message)

    def test_fit_sequences_of_different_length(self):
        lengths = [3, 4, 5]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        # This shouldn't raise
        # ValueError: setting an array element with a sequence.
        h.fit(X, lengths=lengths)

    def test_fit_with_length_one_signal(self):
        lengths = [10, 8, 1]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        # This shouldn't raise
        # ValueError: zero-size array to reduction operation maximum which
        #             has no identity
        h.fit(X, lengths=lengths)

    def test_fit_with_priors(self, params='stmc', n_iter=5):
        startprob_prior = 10 * self.startprob + 2.0
        transmat_prior = 10 * self.transmat + 2.0
        means_prior = self.means
        means_weight = 2.0
        covars_weight = 2.0
        if self.covariance_type in ('full', 'tied'):
            covars_weight += self.n_features
        covars_prior = self.covars[self.covariance_type]

        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        h.startprob_ = self.startprob
        h.startprob_prior = startprob_prior
        h.transmat_ = normalize(
            self.transmat + np.diag(self.prng.rand(self.n_components)), 1)
        h.transmat_prior = transmat_prior
        h.means_ = 20 * self.means
        h.means_prior = means_prior
        h.means_weight = means_weight
        h.covars_ = self.covars[self.covariance_type]
        h.covars_prior = covars_prior
        h.covars_weight = covars_weight

        lengths = [100] * 10
        X, _state_sequence = h.sample(sum(lengths), random_state=self.prng)

        # Re-initialize the parameters and check that we can converge to the
        # original parameter values.
        h_learn = hmm.GaussianHMM(self.n_components, self.covariance_type,
                                  params=params)
        h_learn.n_iter = 0
        h_learn.fit(X, lengths=lengths)

        fit_hmm_and_monitor_log_likelihood(
            h_learn, X, lengths=lengths, n_iter=n_iter)

        # Make sure we've converged to the right parameters.
        # a) means
        self.assertTrue(np.allclose(sorted(h.means_.tolist()),
                                    sorted(h_learn.means_.tolist()),
                                    0.01))
        # b) covars are hard to estimate precisely from a relatively small
        #    sample, thus the large threshold
        self.assertTrue(np.allclose(sorted(h._covars_.tolist()),
                                    sorted(h_learn._covars_.tolist()),
                                    10))


class TestGaussianHMMWithSphericalCovars(GaussianHMMTestMixin, TestCase):
    covariance_type = 'spherical'

    def test_fit_startprob_and_transmat(self):
        self.test_fit('st')


class TestGaussianHMMWithDiagonalCovars(GaussianHMMTestMixin, TestCase):
    covariance_type = 'diag'

    def test_covar_is_writeable(self):
        h = hmm.GaussianHMM(n_components=1, covariance_type="diag",
                            init_params="c")
        X = np.random.normal(size=(1000, 5))
        h._init(X)

        # np.diag returns a read-only view of the array in NumPy 1.9.X.
        # Make sure this doesn't prevent us from fitting an HMM with
        # diagonal covariance matrix. See PR#44 on GitHub for details
        # and discussion.
        assert h._covars_.flags["WRITEABLE"]

    def test_fit_left_right(self):
        transmat = np.zeros((self.n_components, self.n_components))

        # Left-to-right: each state is connected to itself and its
        # direct successor.
        for i in range(self.n_components):
            if i == self.n_components - 1:
                transmat[i, i] = 1.0
            else:
                transmat[i, i] = transmat[i, i + 1] = 0.5

        # Always start in first state
        startprob = np.zeros(self.n_components)
        startprob[0] = 1.0

        lengths = [10, 8, 1]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GaussianHMM(self.n_components, covariance_type="diag",
                            params="mct", init_params="cm")
        h.transmat_ = transmat
        h.startprob_ = startprob
        h.fit(X)

        assert np.allclose(transmat[transmat == 0.0],
                           h.transmat_[transmat == 0.0])


class TestGaussianHMMWithTiedCovars(GaussianHMMTestMixin, TestCase):
    covariance_type = 'tied'


class TestGaussianHMMWithFullCovars(GaussianHMMTestMixin, TestCase):
    covariance_type = 'full'


def create_random_gmm(n_mix, n_features, covariance_type, prng=0):
    prng = check_random_state(prng)
    g = GMM(n_mix, covariance_type=covariance_type)
    g.means_ = prng.randint(-20, 20, (n_mix, n_features))
    mincv = 0.1
    g.covars_ = {
        'spherical': (mincv + mincv * np.dot(prng.rand(n_mix, 1),
                                             np.ones((1, n_features)))) ** 2,
        'tied': (make_spd_matrix(n_features, random_state=prng)
                 + mincv * np.eye(n_features)),
        'diag': (mincv + mincv * prng.rand(n_mix, n_features)) ** 2,
        'full': np.array(
            [make_spd_matrix(n_features, random_state=prng)
             + mincv * np.eye(n_features) for x in range(n_mix)])
    }[covariance_type]
    g.weights_ = normalize(prng.rand(n_mix))
    return g


class GMMHMMTestMixin(object):
    def setUp(self):
        self.prng = np.random.RandomState(9)
        self.n_components = 3
        self.n_mix = 2
        self.n_features = 2
        self.covariance_type = 'diag'
        self.startprob = self.prng.rand(self.n_components)
        self.startprob = self.startprob / self.startprob.sum()
        self.transmat = self.prng.rand(self.n_components, self.n_components)
        self.transmat /= np.tile(self.transmat.sum(axis=1)[:, np.newaxis],
                                 (1, self.n_components))

        self.gmms = []
        for state in range(self.n_components):
            self.gmms.append(create_random_gmm(
                self.n_mix, self.n_features, self.covariance_type,
                prng=self.prng))

    def test_score_samples_and_decode(self):
        h = hmm.GMMHMM(self.n_components)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        h.gmms_ = self.gmms

        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        for g in h.gmms_:
            g.means_ *= 20

        refstateseq = np.repeat(np.arange(self.n_components), 5)
        n_samples = len(refstateseq)
        X = [h.gmms_[x].sample(1, random_state=self.prng).flatten()
             for x in refstateseq]

        _ll, posteriors = h.score_samples(X)

        self.assertEqual(posteriors.shape, (n_samples, self.n_components))
        assert np.allclose(posteriors.sum(axis=1), np.ones(n_samples))

        _logprob, stateseq = h.decode(X)
        assert np.allclose(stateseq, refstateseq)

    def test_sample(self, n=1000):
        h = hmm.GMMHMM(self.n_components, covariance_type=self.covariance_type)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        h.gmms_ = self.gmms
        X, state_sequence = h.sample(n, random_state=self.prng)
        self.assertEqual(X.shape, (n, self.n_features))
        self.assertEqual(len(state_sequence), n)

    @pytest.mark.skip
    def test_fit(self, params='stmwc', n_iter=5, verbose=False, **kwargs):
        h = hmm.GMMHMM(self.n_components, covars_prior=1.0)
        h.startprob_ = self.startprob
        h.transmat_ = normalize(
            self.transmat + np.diag(self.prng.rand(self.n_components)), 1)
        h.gmms_ = self.gmms

        lengths = [10] * 10
        X, _state_sequence = h.sample(sum(lengths), random_state=self.prng)

        # Mess up the parameters and see if we can re-learn them.
        h.n_iter = 0
        h.fit(X, lengths=lengths)
        h.transmat_ = normalize(self.prng.rand(self.n_components,
                                               self.n_components), axis=1)
        h.startprob_ = normalize(self.prng.rand(self.n_components))

        trainll = fit_hmm_and_monitor_log_likelihood(
            h, X, lengths=lengths, n_iter=n_iter)
        if not np.all(np.diff(trainll) > 0) and verbose:
            print('Test train: (%s)\n  %s\n  %s' % (params, trainll,
                                                    np.diff(trainll)))

        # XXX: this test appears to check that training log likelihood should
        # never be decreasing (up to a tolerance of 0.5, why?) but this is not
        # the case when the seed changes.

        self.assertTrue(np.all(np.diff(trainll) > -0.5))

    def test_fit_works_on_sequences_of_different_length(self):
        lengths = [3, 4, 5]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GMMHMM(self.n_components, covariance_type=self.covariance_type)
        # This shouldn't raise
        # ValueError: setting an array element with a sequence.
        h.fit(X, lengths=lengths)


class TestGMMHMMWithDiagCovars(GMMHMMTestMixin, TestCase):
    covariance_type = 'diag'

    def test_fit_startprob_and_transmat(self):
        self.test_fit('st')

    def test_fit_means(self):
        self.test_fit('m')


class TestGMMHMMWithTiedCovars(GMMHMMTestMixin, TestCase):
    covariance_type = 'tied'


class TestGMMHMMWithFullCovars(GMMHMMTestMixin, TestCase):
    covariance_type = 'full'
