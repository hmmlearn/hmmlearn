import numpy as np
import pytest
from sklearn.utils import check_random_state

from hmmlearn import hmm, vhmm
from . import (
    assert_log_likelihood_increasing, compare_variational_and_em_models,
    vi_uniform_startprob_and_transmat)


class TestVariationalCategorical:

    @pytest.fixture(autouse=True)
    def setup(self):
        # We fix the random state here to demonstrate that the model will
        # successfully remove "unnecessary" states.  In practice,
        # one should not set the random_state, and perform multiple
        # training steps, and take the model with the best lower-bound
        self.n_components = 3
        self.implementations = ["scaling", "log"]

    @staticmethod
    def get_beal_models():
        m1 = hmm.CategoricalHMM(3, init_params="")
        m1.n_features = 3
        m1.startprob_ = np.array([1/3., 1/3., 1/3.])
        m1.transmat_ = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        m1.emissionprob_ = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        m2 = hmm.CategoricalHMM(3)
        m2.n_features = 3
        m2.startprob_ = np.array([1/3., 1/3., 1/3.])
        m2.transmat_ = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        m2.emissionprob_ = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        m3 = hmm.CategoricalHMM(1)
        m3.n_features = 3
        m3.startprob_ = np.array([1])
        m3.transmat_ = np.array([[1]])
        m3.emissionprob_ = np.array([[0.5, 0.5, 0]])
        return m1, m2, m3

    @classmethod
    def get_from_one_beal(cls, N, length, rs=None):
        # Just fit the first of the beal models
        model = cls.get_beal_models()[0]
        sequences = []
        lengths = []
        for i in range(N):
            sequences.append(
                model.sample(length, random_state=check_random_state(rs))[0])
            lengths.append(len(sequences[-1]))
        sequences = np.concatenate(sequences)
        return sequences, lengths

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_init_priors(self, implementation):
        sequences, lengths = self.get_from_one_beal(7, 100, None)
        model = vhmm.VariationalCategoricalHMM(
            4, n_iter=500, random_state=1984, init_params="",
            implementation=implementation)
        model.pi_prior_ = np.full((4,), .25)
        model.pi_posterior_ = np.full((4,), 7/4)
        model.transmat_prior_ = np.full((4, 4), .25)
        model.transmat_posterior_ = np.full((4, 4), 7/4)
        model.emissionprob_prior_ = np.full((4, 3), 1/3)
        model.emissionprob_posterior_ = np.asarray([[.3, .4, .3],
                                                    [.8, .1, .1],
                                                    [.2, .2, .6],
                                                    [.2, .6, .2]])
        assert_log_likelihood_increasing(model, sequences, lengths, 10)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_n_features(self, implementation):

        sequences, lengths = self.get_from_one_beal(7, 100, None)
        # Learn n_Features
        model = vhmm.VariationalCategoricalHMM(
            4, implementation=implementation)
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert model.n_features == 3
        # Respect n_features
        model = vhmm.VariationalCategoricalHMM(
            4, implementation=implementation, n_features=5)

        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert model.n_features == 5

        # Too few features
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984,
                implementation=implementation)
            model.n_features = 2
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # No Negative Values
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984,
                implementation=implementation)
            sequences[0] = -1
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Must be integers
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984,
                implementation=implementation)
            sequences = sequences.astype(float)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_init_incorrect_priors(self, implementation):
        sequences, lengths = self.get_from_one_beal(7, 100, None)

        # Test startprob shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="te",
                implementation=implementation)
            model.startprob_prior_ = np.full((3,), .25)
            model.startprob_posterior_ = np.full((4,), 7/4)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="te",
                implementation=implementation)
            model.startprob_prior_ = np.full((4,), .25)
            model.startprob_posterior_ = np.full((3,), 7/4)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Test transmat shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="se",
                implementation=implementation)
            model.transmat_prior_ = np.full((3, 3), .25)
            model.transmat_posterior_ = np.full((4, 4), .25)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="se",
                implementation=implementation)
            model.transmat_prior_ = np.full((4, 4), .25)
            model.transmat_posterior_ = np.full((3, 3), 7/4)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Test emission shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="st",
                implementation=implementation)
            model.emissionprob_prior_ = np.full((3, 3), 1/3)
            model.emissionprob_posterior_ = np.asarray([[.3, .4, .3],
                                                        [.8, .1, .1],
                                                        [.2, .2, .6],
                                                        [.2, .6, .2]])
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Test too many n_features
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="se",
                implementation=implementation)
            model.emissionprob_prior_ = np.full((4, 4), 7/4)
            model.emissionprob_posterior_ = np.full((4, 4), .25)
            model.n_features_ = 10
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Too small n_features
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="se",
                implementation=implementation)
            model.emissionprob_prior_ = np.full((4, 4), 7/4)
            model.emissionprob_posterior_ = np.full((4, 4), .25)

            model.n_features_ = 1
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Test that setting the desired prior value works
        model = vhmm.VariationalCategoricalHMM(
            4, n_iter=500, random_state=1984, init_params="ste",
            implementation=implementation,
            startprob_prior=1, transmat_prior=2, emissionprob_prior=3)
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert np.all(model.startprob_prior_ == 1)
        assert np.all(model.transmat_prior_ == 2)
        assert np.all(model.emissionprob_prior_ == 3)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_beal(self, implementation):
        rs = check_random_state(1984)
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
                                               verbose=False)

        assert_log_likelihood_increasing(model, sequences, lengths, 100)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_and_compare_with_em(self, implementation):
        # Explicitly setting Random State to test that certain
        # model states will become "unused"
        sequences, lengths = self.get_from_one_beal(7, 100, 1984)
        model = vhmm.VariationalCategoricalHMM(
            4, n_iter=500, random_state=1984,
            init_params="e",
            implementation=implementation)
        vi_uniform_startprob_and_transmat(model, lengths)
        model.fit(sequences, lengths)

        # The 1st hidden state will be "unused"
        assert (model.transmat_posterior_[1, :]
                == pytest.approx(.25, rel=1e-3))
        assert (model.emissionprob_posterior_[1, :]
                == pytest.approx(.3333, rel=1e-3))

        # An EM Model should behave the same behavior as a Variational Model,
        # When initialized with the normalized probabilities of the mode of the
        # Variational MOdel.
        em_hmm = hmm.CategoricalHMM(n_components=4, init_params="")
        em_hmm.startprob_ = model.startprob_
        em_hmm.transmat_ = model.transmat_
        em_hmm.emissionprob_ = model.emissionprob_

        compare_variational_and_em_models(model, em_hmm, sequences, lengths)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_length_1_sequences(self, implementation):
        sequences1, lengths1 = self.get_from_one_beal(7, 100, 1984)

        # Include some length 1 sequences
        sequences2, lengths2 = self.get_from_one_beal(1, 1, 1984)
        sequences = np.concatenate([sequences1, sequences2])
        lengths = np.concatenate([lengths1, lengths2])

        model = vhmm.VariationalCategoricalHMM(
            4, n_iter=500, random_state=1984,
            implementation=implementation)
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
