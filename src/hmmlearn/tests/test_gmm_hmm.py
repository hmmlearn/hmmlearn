import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state

from hmmlearn import hmm
from . import assert_log_likelihood_increasing, make_covar_matrix, normalized

pytestmark = pytest.mark.xfail()


def create_random_gmm(n_mix, n_features, covariance_type, prng=0):
    prng = check_random_state(prng)
    g = GaussianMixture(n_mix, covariance_type=covariance_type)
    g.means_ = prng.randint(-20, 20, (n_mix, n_features))
    g.covars_ = make_covar_matrix(covariance_type, n_mix, n_features)
    g.weights_ = normalized(prng.rand(n_mix))
    return g


class GMMHMMTestMixin:
    def setup_method(self, method):
        self.prng = np.random.RandomState(9)
        self.n_components = 3
        self.n_mix = 2
        self.n_features = 2
        self.startprob = normalized(self.prng.rand(self.n_components))
        self.transmat = normalized(
            self.prng.rand(self.n_components, self.n_components), axis=1)
        self.gmms = []
        for state in range(self.n_components):
            self.gmms.append(create_random_gmm(
                self.n_mix, self.n_features, self.covariance_type,
                prng=self.prng))

    def test_score_samples_and_decode(self):
        h = hmm.GMMHMM(self.n_components, covariance_type=self.covariance_type)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        h.gmms_ = self.gmms

        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        for g in h.gmms_:
            g.means_ *= 20

        refstateseq = np.repeat(np.arange(self.n_components), 5)
        n_samples = len(refstateseq)
        X = [h.gmms_[x].sample(1).flatten() for x in refstateseq]

        _ll, posteriors = h.score_samples(X)

        assert posteriors.shape == (n_samples, self.n_components)
        assert_allclose(posteriors.sum(axis=1), np.ones(n_samples))

        _log_prob, stateseq = h.decode(X)
        assert_allclose(stateseq, refstateseq)

    def test_sample(self, n_samples=1000):
        h = hmm.GMMHMM(self.n_components, covariance_type=self.covariance_type)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        h.gmms_ = self.gmms
        X, state_sequence = h.sample(n_samples)
        assert X.shape == (n_samples, self.n_features)
        assert len(state_sequence) == n_samples

    @pytest.mark.parametrize("params", ["stmwc", "wt", "m"])
    def test_fit(self, params, n_iter=5):
        h = hmm.GMMHMM(self.n_components, covariance_type=self.covariance_type,
                       covars_prior=1.0)
        h.startprob_ = self.startprob
        h.transmat_ = normalized(
            self.transmat + np.diag(self.prng.rand(self.n_components)), 1)
        h.gmms_ = self.gmms

        lengths = [10] * 10
        X, _state_sequence = h.sample(sum(lengths), random_state=self.prng)

        # Mess up the parameters and see if we can re-learn them.
        h.n_iter = 0
        h.fit(X, lengths=lengths)
        h.transmat_ = normalized(self.prng.rand(self.n_components,
                                                self.n_components), axis=1)
        h.startprob_ = normalized(self.prng.rand(self.n_components))

        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    def test_fit_works_on_sequences_of_different_length(self):
        lengths = [3, 4, 5]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GMMHMM(self.n_components, covariance_type=self.covariance_type)
        # This shouldn't raise
        # ValueError: setting an array element with a sequence.
        h.fit(X, lengths=lengths)


class TestGMMHMMWithDiagCovars(GMMHMMTestMixin):
    covariance_type = 'diag'


@pytest.mark.xfail
class TestGMMHMMWithTiedCovars(GMMHMMTestMixin):
    covariance_type = 'tied'


@pytest.mark.xfail
class TestGMMHMMWithFullCovars(GMMHMMTestMixin):
    covariance_type = 'full'
