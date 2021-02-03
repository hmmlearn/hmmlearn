import numpy as np
import pytest

from . import log_likelihood_increasing
from . import normalized
from ..hmm import GMMHMM


def sample_from_parallelepiped(low, high, n_samples, random_state):
    (n_features,) = low.shape
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        X[:, i] = random_state.uniform(low[i], high[i], n_samples)
    return X


def prep_params(n_comps, n_mix, n_features, covar_type,
                low, high, random_state):
    # the idea is to generate ``n_comps`` bounding boxes and then
    # generate ``n_mix`` mixture means in each of them

    dim_lims = np.zeros((n_comps + 1, n_features))

    # this generates a sequence of coordinates, which are then used as
    # vertices of bounding boxes for mixtures
    dim_lims[1:] = np.cumsum(
        random_state.uniform(low, high, (n_comps, n_features)), axis=0
    )

    means = np.zeros((n_comps, n_mix, n_features))
    for i, (left, right) in enumerate(zip(dim_lims, dim_lims[1:])):
        means[i] = sample_from_parallelepiped(left, right, n_mix,
                                              random_state)

    startprob = np.zeros(n_comps)
    startprob[0] = 1

    transmat = normalized(random_state.uniform(size=(n_comps, n_comps)),
                          axis=1)

    if covar_type == "spherical":
        covs = random_state.uniform(0.1, 5, size=(n_comps, n_mix))
    elif covar_type == "diag":
        covs = random_state.uniform(0.1, 5, size=(n_comps, n_mix, n_features))
    elif covar_type == "tied":
        covs = np.zeros((n_comps, n_features, n_features))
        for i in range(n_comps):
            low = random_state.uniform(-2, 2, (n_features, n_features))
            covs[i] = np.dot(low.T, low)
    elif covar_type == "full":
        covs = np.zeros((n_comps, n_mix, n_features, n_features))
        for i in range(n_comps):
            for j in range(n_mix):
                low = random_state.uniform(-2, 2,
                                           size=(n_features, n_features))
                covs[i, j] = np.dot(low.T, low)

    weights = normalized(random_state.uniform(size=(n_comps, n_mix)),
                         axis=1)

    return covs, means, startprob, transmat, weights


class GMMHMMTestMixin:
    def setup_method(self, method):
        self.prng = np.random.RandomState(14)
        self.n_components = 3
        self.n_mix = 2
        self.n_features = 2

        self.low, self.high = 10, 15
        (self.covars, self.means,
         self.startprob, self.transmat, self.weights) = prep_params(
            self.n_components, self.n_mix, self.n_features,
            self.covariance_type, self.low, self.high, self.prng
        )

        self.h = GMMHMM(n_components=self.n_components, n_mix=self.n_mix,
                        covariance_type=self.covariance_type,
                        random_state=self.prng)

        self.h.startprob_ = self.startprob
        self.h.transmat_ = self.transmat
        self.h.weights_ = self.weights
        self.h.means_ = self.means
        self.h.covars_ = self.covars

    def test_check_bad_covariance_type(self):
        with pytest.raises(ValueError):
            self.h.covariance_type = "bad_covariance_type"
            self.h._check()

    def test_check_good_covariance_type(self):
        self.h._check()  # should not raise any errors

    def test_sample(self):
        n_samples = 1000
        X, states = self.h.sample(n_samples)
        assert X.shape == (n_samples, self.n_features)
        assert len(states) == n_samples

    def test_init(self):
        n_samples = 1000
        X, _states = self.h.sample(n_samples)
        self.h._init(X)
        self.h._check()  # should not raise any errors

    def test_score_samples_and_decode(self):
        n_samples = 1000
        X, states = self.h.sample(n_samples)

        _ll, posteriors = self.h.score_samples(X)
        assert np.allclose(np.sum(posteriors, axis=1), np.ones(n_samples))

        _viterbi_ll, decoded_states = self.h.decode(X)
        assert np.allclose(states, decoded_states)

    def test_fit(self):
        n_iter = 5
        n_samples = 1000
        lengths = None
        X, _state_sequence = self.h.sample(n_samples)

        # Mess up the parameters and see if we can re-learn them.
        covs0, means0, priors0, trans0, weights0 = prep_params(
            self.n_components, self.n_mix, self.n_features,
            self.covariance_type, self.low, self.high,
            np.random.RandomState(15)
        )
        self.h.covars_ = covs0 * 100
        self.h.means_ = means0
        self.h.startprob_ = priors0
        self.h.transmat_ = trans0
        self.h.weights_ = weights0
        assert log_likelihood_increasing(self.h, X, lengths, n_iter)

    def test_fit_sparse_data(self):
        n_samples = 1000
        self.h.means_ *= 1000  # this will put gaussians very far apart
        X, _states = self.h.sample(n_samples)

        # this should not raise
        # "ValueError: array must not contain infs or NaNs"
        self.h._init(X)
        self.h.fit(X)

    def test_fit_zero_variance(self):
        # Example from issue #2 on GitHub.
        # this data has singular covariance matrix
        X = np.asarray([
            [7.15000000e+02, 5.8500000e+02, 0.00000000e+00, 0.00000000e+00],
            [7.15000000e+02, 5.2000000e+02, 1.04705811e+00, -6.03696289e+01],
            [7.15000000e+02, 4.5500000e+02, 7.20886230e-01, -5.27055664e+01],
            [7.15000000e+02, 3.9000000e+02, -4.57946777e-01, -7.80605469e+01],
            [7.15000000e+02, 3.2500000e+02, -6.43127441e+00, -5.59954834e+01],
            [7.15000000e+02, 2.6000000e+02, -2.90063477e+00, -7.80220947e+01],
            [7.15000000e+02, 1.9500000e+02, 8.45532227e+00, -7.03294373e+01],
            [7.15000000e+02, 1.3000000e+02, 4.09387207e+00, -5.83621216e+01],
            [7.15000000e+02, 6.5000000e+01, -1.21667480e+00, -4.48131409e+01]
        ])

        self.h.fit(X)


class TestGMMHMMWithSphericalCovars(GMMHMMTestMixin):
    covariance_type = 'spherical'


class TestGMMHMMWithDiagCovars(GMMHMMTestMixin):
    covariance_type = 'diag'


class TestGMMHMMWithTiedCovars(GMMHMMTestMixin):
    covariance_type = 'tied'


class TestGMMHMMWithFullCovars(GMMHMMTestMixin):
    covariance_type = 'full'
