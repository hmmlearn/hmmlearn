import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from sklearn.utils import check_random_state

from ..hmm import GMMHMM
from ..vhmm import VariationalGMMHMM, VariationalGaussianHMM
from .test_gmm_hmm import create_random_gmm
from . import (
    assert_log_likelihood_increasing, compare_variational_and_em_models,
    normalized)


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


class GaussianLikeMixin:
    n_components = 3
    n_mix = 1
    n_features = 2
    low, high = 10, 15

    def new_hmm(self, implementation):
        return VariationalGMMHMM(n_components=self.n_components,
                   n_mix=self.n_mix,
                   covariance_type=self.covariance_type,
                   random_state=None,
                   implementation=implementation)

    def new_gaussian(self, implementation):
        return VariationalGaussianHMM(n_components=self.n_components,
                   covariance_type=self.covariance_type,
                   random_state=None,
                   implementation=implementation)

    def new_hmm_to_sample(self, implementation):
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
    def test_learn(self, implementation):
        n_samples = 1000
        h = self.new_hmm_to_sample(implementation)
        X, states = h.sample(n_samples, random_state=32)

        vg = self.new_gaussian(implementation)
        vg.fit(X)
        vh = self.new_hmm(implementation)
        vh.fit(X)
        assert vh.score(X) == pytest.approx(vg.score(X))


class TestVariationalGMMHMMWithFullCovars(GaussianLikeMixin):
    covariance_type = "full"


class TestVariationalGMMHMMWithDiagCovars(GaussianLikeMixin):
    covariance_type = "diag"

# For a Gaussian HMM, Tied covariance means all HMM States share
# one Covariance Matrix. For a GMM HMM, Tied covariance means all
# mixture components within a state share one Covariance Matrix.
# So it does not make sense to compare them two models

# class TestVariationalGMMHMMWithTied(GaussianLikeMixin):
#     covariance_type = "tied"
#

class TestVariationalGMMHMMWithSphericalCovars(GaussianLikeMixin):
    covariance_type = "spherical"


class VariationalGMMHMMTestMixin:
    n_components = 3
    n_mix = 2
    n_features = 2
    low, high = 10, 15

    def new_hmm(self, implementation):
        return VariationalGMMHMM(n_components=self.n_components,
                   n_mix=self.n_mix,
                   covariance_type=self.covariance_type,
                   random_state=None,
                   implementation=implementation, tol=1e-6)

    def new_hmm_to_sample(self, implementation):
        prng = np.random.RandomState(44)
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

    #@pytest.mark.parametrize("implementation", ["scaling", "log"])
    #def test_check_good_covariance_type(self, implementation):
    #    h = self.new_hmm(implementation)
    #    h._check()  # should not raise any errors

    def do_test_learn(self, implementation, X, lengths):
        vb_hmm = self.new_hmm(implementation)
        vb_hmm.fit(X, lengths)
        assert not np.any(np.isnan(vb_hmm.means_posterior_))

        em_hmm = GMMHMM(
            n_components=vb_hmm.n_components,
            n_mix=vb_hmm.n_mix,
            implementation=implementation,
            covariance_type=self.covariance_type,
        )
        em_hmm.startprob_ = vb_hmm.startprob_
        em_hmm.transmat_ = vb_hmm.transmat_
        em_hmm.weights_ = vb_hmm.weights_
        em_hmm.means_ = vb_hmm.means_
        em_hmm.covars_ = vb_hmm.covars_
        compare_variational_and_em_models(vb_hmm, em_hmm, X, lengths)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_learn(self, implementation):
        n_samples = 2000
        source = self.new_hmm_to_sample(implementation)
        X, states = source.sample(n_samples, random_state=32)
        self.do_test_learn(implementation, X, [X.shape[0]])

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_learn_multisequence(self, implementation):
        n_samples = 2000
        source = self.new_hmm_to_sample(implementation)
        X, states = source.sample(n_samples, random_state=32)
        self.do_test_learn(implementation, X, [n_samples //4] * 4)

#    @pytest.mark.parametrize("implementation", ["scaling", "log"])
#    def test_sample(self, implementation):
#        n_samples = 1000
#        h = self.new_hmm(implementation)
#        X, states = h.sample(n_samples)
#        assert X.shape == (n_samples, self.n_features)
#        assert len(states) == n_samples
#
#    @pytest.mark.parametrize("implementation", ["scaling", "log"])
#    def test_init(self, implementation):
#        n_samples = 1000
#        h = self.new_hmm(implementation)
#        X, _states = h.sample(n_samples)
#        h._init(X, [n_samples])
#        h._check()  # should not raise any errors
#
#    @pytest.mark.parametrize("implementation", ["scaling", "log"])
#    def test_score_samples_and_decode(self, implementation):
#        n_samples = 1000
#        h = self.new_hmm(implementation)
#        X, states = h.sample(n_samples)
#
#        _ll, posteriors = h.score_samples(X)
#        assert_allclose(np.sum(posteriors, axis=1), np.ones(n_samples))
#
#        _viterbi_ll, decoded_states = h.decode(X)
#        assert_allclose(states, decoded_states)
#
    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_sparse_data(self, implementation):
        n_samples = 1000
        h = self.new_hmm_to_sample(implementation)
        h.means_ *= 1000  # this will put gaussians very far apart
        X, _states = h.sample(n_samples)

        m = self.new_hmm_to_sample(implementation)
        # this should not raise
        # "ValueError: array must not contain infs or NaNs"
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


class TestVariationalGMMHMMWithSphericalCovars(VariationalGMMHMMTestMixin):
    covariance_type = 'spherical'

class TestVariationalGMMHMMWithDiagCovars(VariationalGMMHMMTestMixin):
    covariance_type = 'diag'

class TestVariationalGMMHMMWithFullCovars(VariationalGMMHMMTestMixin):
    covariance_type = 'full'

class TestVariationalGMMHMMWithTiedCovars(VariationalGMMHMMTestMixin):
    covariance_type = 'tied'
#
