import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from sklearn.utils import check_random_state

from ..hmm import GMMHMM
from .test_gmm_hmm import create_random_gmm
from . import assert_log_likelihood_increasing, normalized


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
            covs[i] = low.T @ low
    elif covar_type == "full":
        covs = np.zeros((n_comps, n_mix, n_features, n_features))
        for i in range(n_comps):
            for j in range(n_mix):
                low = random_state.uniform(-2, 2,
                                           size=(n_features, n_features))
                covs[i, j] = low.T @ low

    weights = normalized(random_state.uniform(size=(n_comps, n_mix)), axis=1)

    return covs, means, startprob, transmat, weights


class GMMHMMTestMixin:
    n_components = 3
    n_mix = 2
    n_features = 2
    low, high = 10, 15

    def new_hmm(self, implementation):
        prng = np.random.RandomState(14)
        covars, means, startprob, transmat, weights = prep_params(
            self.n_components, self.n_mix, self.n_features,
            self.covariance_type, self.low, self.high, prng)
        h = GMMHMM(n_components=self.n_components, n_mix=self.n_mix,
                   covariance_type=self.covariance_type,
                   random_state=prng,
                   implementation=implementation)
        h.startprob_ = startprob
        h.transmat_ = transmat
        h.weights_ = weights
        h.means_ = means
        h.covars_ = covars
        return h

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_check_bad_covariance_type(self, implementation):
        h = self.new_hmm(implementation)
        with pytest.raises(ValueError):
            h.covariance_type = "bad_covariance_type"
            h._check()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_check_good_covariance_type(self, implementation):
        h = self.new_hmm(implementation)
        h._check()  # should not raise any errors

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_sample(self, implementation):
        n_samples = 1000
        h = self.new_hmm(implementation)
        X, states = h.sample(n_samples)
        assert X.shape == (n_samples, self.n_features)
        assert len(states) == n_samples

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_init(self, implementation):
        n_samples = 1000
        h = self.new_hmm(implementation)
        X, _states = h.sample(n_samples)
        h._init(X, [n_samples])
        h._check()  # should not raise any errors

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_score_samples_and_decode(self, implementation):
        n_samples = 1000
        h = self.new_hmm(implementation)
        X, states = h.sample(n_samples)

        _ll, posteriors = h.score_samples(X)
        assert_allclose(np.sum(posteriors, axis=1), np.ones(n_samples))

        _viterbi_ll, decoded_states = h.decode(X)
        assert_allclose(states, decoded_states)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit(self, implementation):
        n_iter = 5
        n_samples = 1000
        lengths = None
        h = self.new_hmm(implementation)
        X, _state_sequence = h.sample(n_samples)

        # Mess up the parameters and see if we can re-learn them.
        covs0, means0, priors0, trans0, weights0 = prep_params(
            self.n_components, self.n_mix, self.n_features,
            self.covariance_type, self.low, self.high,
            np.random.RandomState(15)
        )
        h.covars_ = covs0 * 100
        h.means_ = means0
        h.startprob_ = priors0
        h.transmat_ = trans0
        h.weights_ = weights0
        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_sparse_data(self, implementation):
        n_samples = 1000
        h = self.new_hmm(implementation)
        h.means_ *= 1000  # this will put gaussians very far apart
        X, _states = h.sample(n_samples)

        # this should not raise
        # "ValueError: array must not contain infs or NaNs"
        h._init(X, [1000])
        h.fit(X)

    @pytest.mark.xfail
    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_zero_variance(self, implementation):
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

        h = self.new_hmm(implementation)
        h.fit(X)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_criterion(self, implementation):
        random_state = check_random_state(2013)
        m1 = self.new_hmm(implementation)
        # Spread the means out to make this easier
        m1.means_ *= 10

        X, _ = m1.sample(4000, random_state=random_state)
        aic = []
        bic = []
        ns = [2, 3, 4, 5]
        for n in ns:
            h = GMMHMM(n, n_mix=2, covariance_type=self.covariance_type,
                random_state=random_state, implementation=implementation)
            h.fit(X)
            aic.append(h.aic(X))
            bic.append(h.bic(X))

        assert np.all(aic) > 0
        assert np.all(bic) > 0
        # AIC / BIC pick the right model occasionally
        # assert ns[np.argmin(aic)] == self.n_components
        # assert ns[np.argmin(bic)] == self.n_components


class TestGMMHMMWithSphericalCovars(GMMHMMTestMixin):
    covariance_type = 'spherical'


class TestGMMHMMWithDiagCovars(GMMHMMTestMixin):
    covariance_type = 'diag'


class TestGMMHMMWithTiedCovars(GMMHMMTestMixin):
    covariance_type = 'tied'


class TestGMMHMMWithFullCovars(GMMHMMTestMixin):
    covariance_type = 'full'


class TestGMMHMM_KmeansInit:
    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_kmeans(self, implementation):
        # Generate two isolated cluster.
        # The second cluster has no. of points less than n_mix.
        np.random.seed(0)
        data1 = np.random.uniform(low=0, high=1, size=(100, 2))
        data2 = np.random.uniform(low=5, high=6, size=(5, 2))
        data = np.r_[data1, data2]
        model = GMMHMM(n_components=2, n_mix=10, n_iter=5,
                       implementation=implementation)
        model.fit(data)  # _init() should not fail here
        # test whether the means are bounded by the data lower- and upperbounds
        assert_array_less(0, model.means_)
        assert_array_less(model.means_, 6)


class TestGMMHMM_MultiSequence:

    @pytest.mark.parametrize("covtype",
                             ["diag", "spherical", "tied", "full"])
    def test_chunked(sellf, covtype, init_params='mcw'):
        np.random.seed(0)
        gmm = create_random_gmm(3, 2, covariance_type=covtype, prng=0)
        gmm.covariances_ = gmm.covars_
        data = gmm.sample(n_samples=1000)[0]

        model1 = GMMHMM(n_components=3, n_mix=2, covariance_type=covtype,
                        random_state=1, init_params=init_params)
        model2 = GMMHMM(n_components=3, n_mix=2, covariance_type=covtype,
                        random_state=1, init_params=init_params)
        # don't use random parameters for testing
        init = 1. / model1.n_components
        for model in (model1, model2):
            model.startprob_ = np.full(model.n_components, init)
            model.transmat_ = \
                np.full((model.n_components, model.n_components), init)

        model1.fit(data)
        model2.fit(data, lengths=[200] * 5)

        assert_allclose(model1.means_, model2.means_, rtol=0, atol=1e-2)
        assert_allclose(model1.covars_, model2.covars_, rtol=0, atol=1e-3)
        assert_allclose(model1.weights_, model2.weights_, rtol=0, atol=1e-3)
        assert_allclose(model1.transmat_, model2.transmat_, rtol=0, atol=1e-2)
