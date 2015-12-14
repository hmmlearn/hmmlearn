from __future__ import absolute_import

import numpy as np
from sklearn.datasets.samples_generator import make_spd_matrix
from sklearn.mixture import GMM
from sklearn.utils import check_random_state

from hmmlearn import hmm
from hmmlearn.utils import normalize

from ._test_common import fit_hmm_and_monitor_log_likelihood


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
    def setup_method(self, method):
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

        assert posteriors.shape == (n_samples, self.n_components)
        assert np.allclose(posteriors.sum(axis=1), np.ones(n_samples))

        _logprob, stateseq = h.decode(X)
        assert np.allclose(stateseq, refstateseq)

    def test_sample(self, n=1000):
        h = hmm.GMMHMM(self.n_components, covariance_type=self.covariance_type)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        h.gmms_ = self.gmms
        X, state_sequence = h.sample(n, random_state=self.prng)
        assert X.shape == (n, self.n_features)
        assert len(state_sequence) == n

    def test_fit(self, params='stmwc', n_iter=5):
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
        assert np.all(np.diff(trainll) <= 0)

    def test_fit_works_on_sequences_of_different_length(self):
        lengths = [3, 4, 5]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GMMHMM(self.n_components, covariance_type=self.covariance_type)
        # This shouldn't raise
        # ValueError: setting an array element with a sequence.
        h.fit(X, lengths=lengths)


class TestGMMHMMWithDiagCovars(GMMHMMTestMixin):
    covariance_type = 'diag'

    def test_fit_startprob_and_transmat(self):
        self.test_fit('st')

    def test_fit_means(self):
        self.test_fit('m')


class TestGMMHMMWithTiedCovars(GMMHMMTestMixin):
    covariance_type = 'tied'


class TestGMMHMMWithFullCovars(GMMHMMTestMixin):
    covariance_type = 'full'
