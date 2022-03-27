import numpy as np
import pytest
import scipy.stats

from sklearn.utils import check_random_state

from hmmlearn import hmm, vhmm

from . import assert_log_likelihood_increasing, normalized

class TestVariationalGaussian:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prng = prng = np.random.RandomState(32)
        self.n_components = n_components = 3

        self.implementations = ["scaling", "log"]

    def get_mcgrory_titterington(self):
        m1 = hmm.GaussianHMM(3, init_params="")
        m1.n_features = 3
        m1.startprob_ = np.array([1/3., 1/3., 1/3.])
        m1.transmat_ = np.array([[0.15, 0.8, 0.05], [0.5, 0.1, 0.4], [0.3, 0.4, 0.3]])
        m1.means_ = np.array([[1], [2], [3]])
        m1.covars_  = np.array([[0.25], [0.1], [0.7]])
        return m1

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_mcgrory_titterington(self, implementation):

        random_state = check_random_state(1984)

        model = self.get_mcgrory_titterington()
        sequences = model.sample(500, random_state=random_state)[0]
        lengths = [500]
        model = vhmm.VariationalGaussianHMM(5, n_iter=500,
                                            implementation=implementation,
                                            tol=1e-6,
                                            random_state=random_state,
                                            verbose=False)
        model.fit(sequences, lengths)
        print(model.monitor_.history)
        print(model.startprob_posterior_)
        print(model.transmat_posterior_)
        print(model.means_posterior_)
        print(model.covars_posterior_)
