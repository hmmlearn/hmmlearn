import numpy as np
import pytest
import scipy.stats

from hmmlearn import vhmm
from . import assert_log_likelihood_increasing, normalized

class TestVariationalCategorical:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prng = prng = np.random.RandomState(32)
        self.n_components = n_components = 3

        self.implementations = ["log", "scaling"]

    # def test_fit1(self, params='str', n_iter=5):

    #     model = vhmm.VariationalCategoricalHMM(12)
    #     sequences = []
    #     lengths = []
    #     sequences.append([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    #     sequences.append([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    #     sequences.append([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    #     lengths += [12, 12, 12]
    #     sequences.append([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2])
    #     sequences.append([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2])
    #     sequences.append([1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2])
    #     lengths += [12, 12, 12]
    #     sequences.append([2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
    #     sequences.append([2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
    #     sequences.append([2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
    #     lengths += [12, 12, 12]

    #     sequences = np.concatenate(sequences)[:, None]
    #     model.fit(sequences, lengths)

    #     print(model.transmat_posterior_ / model.transmat_posterior_.sum(axis=1))
    def test_fit2(self, params='str', n_iter=5):

        for implementation in self.implementations:
            model = vhmm.VariationalCategoricalHMM(4, implementation=implementation, tol=1e-6, random_state=23)
        sequences = []
        lengths = []
        sequences.append([0, 1, 2, 0, 1, 4, 0, 1, 2, 0, 1, 4])
        sequences.append([0, 1, 2, 0, 1, 4, 0, 1, 2, 0, 1, 4])
        sequences.append([0, 1, 2, 0, 1, 4, 0, 1, 2, 0, 1, 4])
        lengths += [12, 12, 12]

        sequences = np.concatenate(sequences)[:, None]
        model.fit(sequences, lengths)

        print(model.transmat_posterior_ / model.transmat_posterior_.sum(axis=1)[:, None])
        print(model.transmat_posterior_)
        print(model.emissions_posterior_)
