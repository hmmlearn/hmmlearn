import numpy as np
import pytest
import scipy.stats

from sklearn.utils import check_random_state

from hmmlearn import hmm, vhmm

from . import assert_log_likelihood_increasing, normalized, \
        compare_variational_and_em_models

class TestVariationalCategorical:

    @pytest.fixture(autouse=True)
    def setup(self):
        # We fix the random state here to demonstrate that the model will
        # successfully remove "unnecessary" states.  In practice,
        # one should not set the random_state, and perform multiple
        # training steps, and take the model with the best lower-bound

        self.prng = check_random_state(1984)
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
        rs = self.prng
        m1, m2, m3 = self.get_beal_models()
        sequences = []
        lengths = []
        for i in range(7):
            for m in [m1, m2, m3]:
                sequences.append(m.sample(39, random_state=rs)[0])
                lengths.append(len(sequences[-1]))
        sequences = np.concatenate(sequences)
        model = vhmm.VariationalCategoricalHMM(12, n_iter=500,
                                               implementation=implementation,
                                               tol=1e-6,
                                               random_state=rs,
                                               verbose=True)
        model.fit(sequences, lengths)
        print(model.monitor_.history)
        print(model.startprob_posterior_)
        print(model.transmat_posterior_)
        print(model.emissions_posterior_)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_simple(self, implementation):

        model = self.get_beal_models()[0]
        sequences = []
        lengths = []
        for i in range(7):
            sequences.append(model.sample(39, random_state=self.prng)[0])
            lengths.append(len(sequences[-1]))

        sequences = np.concatenate(sequences)
        model = vhmm.VariationalCategoricalHMM(4, n_iter=500,
                                               implementation=implementation,
                                               random_state=self.prng)

        model.fit(sequences, lengths)

        # print(model.monitor_.history)
        # print(model.startprob_posterior_)
        # print(model.transmat_posterior_)
        # print(model.emissions_posterior_)

        # The 1st hidden state will be "unused"
        check = model.transmat_posterior_[0, :] == pytest.approx(.25, rel=1e-3)
        assert np.all(check)
        check = model.emissions_posterior_[0, :] == pytest.approx(.3333,
                                                                  rel=1e-3)
        assert np.all(check)

        # An EM Model should behave the same as a Variational Model,
        # When initialized with the normalized probabilities of the mode of the
        # Variational MOdel.
        em_hmm = hmm.MultinomialHMM(n_components=4, init_params="")
        em_hmm.startprob_ = model.startprob_normalized_
        em_hmm.transmat_ = model.transmat_normalized_
        em_hmm.emissionprob_ = model.emissions_normalized_

        compare_variational_and_em_models(model, em_hmm, sequences, lengths)

