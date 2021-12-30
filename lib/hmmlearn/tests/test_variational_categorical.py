import numpy as np
import pytest
import scipy.stats

from sklearn.utils import check_random_state

from hmmlearn import hmm, vhmm

from . import assert_log_likelihood_increasing, normalized

class TestVariationalCategorical:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prng = prng = np.random.RandomState(32)
        self.n_components = n_components = 3

        self.implementations = ["scaling", "log"]

    def get_beal_models(self):
        m1 = hmm.MultinomialHMM(3, init_params="")
        m1.n_features = 3
        m1.startprob_ = np.array([1/3., 1/3., 1/3.])
        m1.transmat_ = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        m1.emissionprob_ = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        m2 = hmm.MultinomialHMM(3)
        m2.n_features = 3
        m2.startprob_ = np.array([1/3., 1/3., 1/3.])
        m2.transmat_ = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        m2.emissionprob_ = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        m3 = hmm.MultinomialHMM(1)
        m3.n_features = 3
        m3.startprob_ = np.array([1])
        m3.transmat_ = np.array([[1]])
        m3.emissionprob_ = np.array([[0.5, 0.5, 0]])
        return m1, m2, m3


    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_beal(self, implementation):

        random_state = check_random_state(1984)

        m1, m2, m3 = self.get_beal_models()
        sequences = []
        lengths = []
        for i in range(7):
            for m in [m1, m2, m3]:
                sequences.append(m.sample(39, random_state=random_state)[0])
                lengths.append(len(sequences[-1]))
        sequences = np.concatenate(sequences)
        model = vhmm.VariationalCategoricalHMM(12, n_iter=500,
                                               implementation=implementation,
                                               tol=1e-6,
                                               random_state=random_state,
                                               verbose=False)
        model.fit(sequences, lengths)
        print(model.monitor_.history)
        print(model.startprob_posterior_)
        print(model.transmat_posterior_)
        print(model.emissions_posterior_)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    @pytest.mark.parametrize("decode_algo", ["viterbi", "map"])
    def test_fit_simple(self, implementation, decode_algo):

        model = self.get_beal_models()[0]
        sequences = []
        lengths = []
        for i in range(7):
            sequences.append(model.sample(39, random_state=33)[0])
            lengths.append(len(sequences[-1]))

        sequences = np.concatenate(sequences)
        model = vhmm.VariationalCategoricalHMM(4, n_iter=500,
                                               implementation=implementation,
                                               random_state=43)

        model.fit(sequences, lengths)

        # The 3rd hidden state will be "unused"
        check = model.transmat_posterior_[2, :] == pytest.approx(.25, rel=1e-3)
        assert np.all(check)
        check = model.emissions_posterior_[2, :] == pytest.approx(.3333,
                                                                  rel=1e-3)
        assert np.all(check)

        # An EM Model should behave the same as a Variational Model,
        # When initialized with the normalized probabilities of the mode of the
        # Variational MOdel.
        em_hmm = hmm.MultinomialHMM(n_components=4, init_params="")
        em_hmm.startprob_ = model.startprob_normalized_
        em_hmm.transmat_ = model.transmat_normalized_
        em_hmm.emissionprob_ = model.emissions_normalized_

        em_score = em_hmm.score(sequences, lengths)
        vi_score = model.score(sequences, lengths)
        em_scores = em_hmm.predict(sequences, lengths)
        vi_scores = model.predict(sequences, lengths)
        assert em_score == pytest.approx(vi_score)
        assert np.all(em_scores == vi_scores)

        em_logprob, em_path = em_hmm.decode(sequences, lengths,
                                            algorithm=decode_algo)
        vi_logprob, vi_path = model.decode(sequences, lengths,
                                           algorithm=decode_algo)
        assert em_logprob == pytest.approx(vi_logprob)
        assert np.all(em_path == vi_path)

        em_predict = em_hmm.predict(sequences, lengths)
        vi_predict = model.predict(sequences, lengths)
        assert np.all(em_predict == vi_predict)
        em_logprob, em_posteriors = em_hmm.score_samples(sequences, lengths)
        vi_logprob, vi_posteriors = model.score_samples(sequences, lengths)
        assert em_logprob == pytest.approx(vi_logprob), implementation
        assert np.all(em_posteriors == pytest.approx(vi_posteriors))

        em_obs, em_states = em_hmm.sample(100, random_state=42)
        vi_obs, vi_states = model.sample(100, random_state=42)
        assert np.all(em_obs == vi_obs)
        assert np.all(em_states == vi_states)
