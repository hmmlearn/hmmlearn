import numpy as np
import pandas as pd
import pytest
import scipy.stats

from sklearn.utils import check_random_state

from hmmlearn import vhmm
from . import assert_log_likelihood_increasing, normalized

class TestVariationalCategorical:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prng = prng = np.random.RandomState(32)
        self.n_components = n_components = 3

        self.implementations = ["log", "scaling"]

    def test_fit_beal(self, params='str', n_iter=5):

        random_state = check_random_state(2022)

        sequences = []
        lengths = []
        for i in range(10):
            sequences.append([0, 1, 2] * 20)
            lengths.append(len(sequences[-1]))
            sequences.append([0, 2, 1] * 20)
            lengths.append(len(sequences[-1]))
            sequences.append(random_state.random_integers(0, 1, 60))
            lengths.append(len(sequences[-1]))

        sequences = np.concatenate(sequences)[:, None]
        for implementation in self.implementations:
            model = vhmm.VariationalCategoricalHMM(9, n_iter=500,  implementation=implementation, tol=1e-6, random_state=2001)

            model.fit(sequences, lengths)

            print(model.transmat_posterior_ / model.transmat_posterior_.sum(axis=1)[:, None])
            print(pd.DataFrame(model.transmat_posterior_))
            print(pd.DataFrame(model.emissions_posterior_))

    def test_fit_simple(self, params='str', n_iter=5):

        random_state = check_random_state(2022)

        sequences = []
        lengths = []
        for i in range(10):
            sequences.append([0, 1, 2] * 20)
            lengths.append(len(sequences[-1]))

        sequences = np.concatenate(sequences)[:, None]
        for implementation in self.implementations:
            model = vhmm.VariationalCategoricalHMM(4, n_iter=500,  implementation=implementation, tol=1e-6, random_state=2001)

            model.fit(sequences, lengths)

            print(model.transmat_posterior_ / model.transmat_posterior_.sum(axis=1)[:, None])
            print(pd.DataFrame(model.transmat_posterior_))
            print(pd.DataFrame(model.emissions_posterior_))
