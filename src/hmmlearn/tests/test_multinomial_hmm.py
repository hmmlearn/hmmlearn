import numpy as np
from numpy.testing import assert_allclose
import pytest

from hmmlearn import hmm

from . import assert_log_likelihood_increasing, normalized


class TestMultinomialHMM:
    n_components = 2
    n_features = 4
    n_trials = 5

    def new_hmm(self, impl):
        h = hmm.MultinomialHMM(
            n_components=self.n_components,
            n_trials=self.n_trials,
            implementation=impl)
        h.startprob_ = np.array([.6, .4])
        h.transmat_ = np.array([[.8, .2], [.2, .8]])
        h.emissionprob_ = np.array([[.5, .3, .1, .1], [.1, .1, .4, .4]])
        return h

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_attributes(self, implementation):
        with pytest.raises(ValueError):
            h = self.new_hmm(implementation)
            h.emissionprob_ = []
            h._check()
        with pytest.raises(ValueError):
            h.emissionprob_ = np.zeros((self.n_components - 2,
                                        self.n_features))
            h._check()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_score_samples(self, implementation):
        X = np.array([
            [1, 1, 3, 0],
            [3, 1, 1, 0],
            [3, 0, 2, 0],
            [2, 2, 0, 1],
            [2, 2, 0, 1],
            [0, 1, 1, 3],
            [1, 0, 3, 1],
            [2, 0, 1, 2],
            [0, 2, 1, 2],
            [1, 0, 1, 3],
        ])
        n_samples = X.shape[0]
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
        assert len(np.unique(X)) == self.n_trials + 1
        assert (X.sum(axis=1) == self.n_trials).all()
        h.n_trials = None
        with pytest.raises(ValueError):
            h.sample(n_samples)
        h.n_trials = [1, 2, 3]
        with pytest.raises(ValueError):
            h.sample(n_samples)

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
        # Also mess up trial counts.
        h.n_trials = None
        X[::2] *= 2

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
        h = hmm.MultinomialHMM(
            n_components=self.n_components, n_trials=self.n_trials,
            params=params, init_params=params)
        h._init(X, lengths)

        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test__check_and_set_multinomial_n_features_n_trials(
            self, implementation):
        h = hmm.MultinomialHMM(
            n_components=2, n_trials=None, implementation=implementation)
        h._check_and_set_n_features(
            np.array([[0, 2, 3, 0], [1, 0, 2, 2]]))
        assert (h.n_trials == 5).all()
        with pytest.raises(ValueError):  # wrong dimensions
            h._check_and_set_n_features(
                np.array([[0, 0, 2, 1, 3, 1, 1]]))
        with pytest.raises(ValueError):  # not added up to n_trials
            h._check_and_set_n_features(
                np.array([[0, 0, 1, 1], [3, 1, 1, 0]]))
        with pytest.raises(ValueError):  # non-integral
            h._check_and_set_n_features(
                np.array([[0., 2., 0., 3.], [0.0, 2.5, 2.5, 0.0]]))
        with pytest.raises(ValueError):  # negative integers
            h._check_and_set_n_features(
                np.array([[0, -2, 1, 6], [5, 6, -6, 0]]))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_compare_with_categorical_hmm(self, implementation):
        n_components = 2   # ['Rainy', 'Sunny']
        n_features = 3     # ['walk', 'shop', 'clean']
        n_trials = 1
        startprob = np.array([0.6, 0.4])
        transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
        emissionprob = np.array([[0.1, 0.4, 0.5],
                                 [0.6, 0.3, 0.1]])
        h1 = hmm.MultinomialHMM(
            n_components=n_components, n_trials=n_trials,
            implementation=implementation)
        h2 = hmm.CategoricalHMM(
            n_components=n_components, implementation=implementation)

        h1.startprob_ = startprob
        h2.startprob_ = startprob

        h1.transmat_ = transmat
        h2.transmat_ = transmat

        h1.emissionprob_ = emissionprob
        h2.emissionprob_ = emissionprob

        X1 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
        X2 = [[0], [1], [2]]  # different input format for CategoricalHMM
        log_prob1, state_sequence1 = h1.decode(X1, algorithm="viterbi")
        log_prob2, state_sequence2 = h2.decode(X2, algorithm="viterbi")
        assert round(np.exp(log_prob1), 5) == 0.01344
        assert round(np.exp(log_prob2), 5) == 0.01344

        assert_allclose(state_sequence1, [1, 0, 0])
        assert_allclose(state_sequence2, [1, 0, 0])

        posteriors1 = h1.predict_proba(X1)
        assert_allclose(posteriors1, [
            [0.23170303, 0.76829697],
            [0.62406281, 0.37593719],
            [0.86397706, 0.13602294],
        ], rtol=0, atol=1e-6)

        posteriors2 = h2.predict_proba(X2)
        assert_allclose(posteriors2, [
            [0.23170303, 0.76829697],
            [0.62406281, 0.37593719],
            [0.86397706, 0.13602294],
        ], rtol=0, atol=1e-6)
