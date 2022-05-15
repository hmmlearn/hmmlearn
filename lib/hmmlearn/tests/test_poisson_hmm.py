import numpy as np
from numpy.testing import assert_allclose
import pytest

from hmmlearn import hmm

from . import assert_log_likelihood_increasing, normalized


class TestPoissonHMM:

    n_components = 2
    n_features = 3

    def new_hmm(self, impl):
        h = hmm.PoissonHMM(self.n_components, implementation=impl)
        h.startprob_ = np.array([0.6, 0.4])
        h.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        h.lambdas_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
        return h

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_attributes(self, implementation):
        with pytest.raises(ValueError):
            h = self.new_hmm(implementation)
            h.lambdas_ = []
            h._check()
        with pytest.raises(ValueError):
            h.lambdas_ = np.zeros((self.n_components - 2, self.n_features))
            h._check()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_score_samples(self, implementation, n_samples=10):
        rng = np.random.default_rng(0)
        lambdas = rng.random(size=(self.n_components, self.n_features)) * 5
        h = self.new_hmm(implementation)

        ll, posteriors = h.score_samples(X)
        assert posteriors.shape == (n_samples, self.n_components)
        assert_allclose(posteriors.sum(axis=1), np.ones(n_samples))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_sample(self, implementation, n_samples=1000):
        h = self.new_hmm(implementation)
        X, state_sequence = h.sample(n_samples)
        assert X.ndim == 2
        assert len(X) == len(state_sequence) == n_samples
        assert len(np.unique(X)) == self.n_features

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit(self, implementation, params='ste', n_iter=5):

        h = self.new_hmm(implementation)
        h.params = params

        lengths = np.array([10] * 10)
        X, _state_sequence = h.sample(lengths.sum())

        # Mess up the parameters and see if we can re-learn them.
        h.startprob_ = normalized(np.random.random(self.n_components))
        h.transmat_ = normalized(
            np.random.random((self.n_components, self.n_components)),
            axis=1)
        h.emissionprob_ = normalized(
            np.random.random((self.n_components, self.n_features)),
            axis=1)

        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_emissionprob(self, implementation):
        self.test_fit(implementation, 'e')

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_with_init(self, implementation, params='ste', n_iter=5):
        lengths = [10] * 10
        h = self.new_hmm(implementation)
        X, _state_sequence = h.sample(sum(lengths))

        # use init_function to initialize paramerters
        h = hmm.MultinomialHMM(self.n_components, params=params,
                               init_params=params)
        h._init(X)

        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test__check_and_set_multinomial_n_features(self, implementation):
        h = self.new_hmm(implementation)
        h._check_and_set_multinomial_n_features(
            np.array([[0, 0, 2, 1, 3, 1, 1]]))
        h._check_and_set_multinomial_n_features(
            np.array([[0, 0, 1, 3, 1]], np.uint8))
        with pytest.raises(ValueError):  # non-integral
            h._check_and_set_multinomial_n_features(
                np.array([[0., 2., 1., 3.]]))
        with pytest.raises(ValueError):  # negative integers
            h._check_and_set_multinomial_n_features(
                np.array([[0, -2, 1, 3, 1, 1]]))
