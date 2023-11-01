import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.utils import check_random_state

from hmmlearn import hmm

from . import assert_log_likelihood_increasing, normalized


class TestPoissonHMM:
    n_components = 2
    n_features = 3

    def new_hmm(self, impl):
        h = hmm.PoissonHMM(self.n_components, implementation=impl,
                           random_state=0)
        h.startprob_ = np.array([0.6, 0.4])
        h.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        h.lambdas_ = np.array([[3.1, 1.4, 4.5], [1.6, 5.3, 0.1]])
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
    def test_score_samples(self, implementation, n_samples=1000):
        h = self.new_hmm(implementation)
        X, state_sequence = h.sample(n_samples)
        assert X.ndim == 2
        assert len(X) == len(state_sequence) == n_samples

        ll, posteriors = h.score_samples(X)
        assert posteriors.shape == (n_samples, self.n_components)
        assert_allclose(posteriors.sum(axis=1), np.ones(n_samples))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit(self, implementation, params='stl', n_iter=5):

        h = self.new_hmm(implementation)
        h.params = params

        lengths = np.array([10] * 10)
        X, _state_sequence = h.sample(lengths.sum())

        # Mess up the parameters and see if we can re-learn them.
        np.random.seed(0)
        h.startprob_ = normalized(np.random.random(self.n_components))
        h.transmat_ = normalized(
            np.random.random((self.n_components, self.n_components)),
            axis=1)
        h.lambdas_ = np.random.gamma(
            shape=2, size=(self.n_components, self.n_features))

        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_lambdas(self, implementation):
        self.test_fit(implementation, 'l')

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_with_init(self, implementation, params='stl', n_iter=5):
        lengths = [10] * 10
        h = self.new_hmm(implementation)
        X, _state_sequence = h.sample(sum(lengths))

        # use init_function to initialize paramerters
        h = hmm.PoissonHMM(self.n_components, params=params,
                           init_params=params)
        h._init(X, lengths)

        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_criterion(self, implementation):
        random_state = check_random_state(412)
        m1 = self.new_hmm(implementation)
        X, _ = m1.sample(2000, random_state=random_state)

        aic = []
        bic = []
        ns = [2, 3, 4]
        for n in ns:
            h = hmm.PoissonHMM(n, n_iter=500,
                random_state=random_state, implementation=implementation)
            h.fit(X)
            aic.append(h.aic(X))
            bic.append(h.bic(X))

        assert np.all(aic) > 0
        assert np.all(bic) > 0
        # AIC / BIC pick the right model occasionally
        # assert ns[np.argmin(aic)] == 2
        # assert ns[np.argmin(bic)] == 2
