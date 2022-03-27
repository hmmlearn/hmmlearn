import numpy as np
import pytest
import scipy.stats

from sklearn.utils import check_random_state

from hmmlearn import hmm, vhmm

from . import assert_log_likelihood_increasing, normalized, \
        compare_variational_and_em_models

class TestVariationalGaussian:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prng = prng = np.random.RandomState(32)
        self.n_components = n_components = 3

        self.implementations = ["scaling", "log"]

    def get_mcgrory_titterington(self):
        m1 = hmm.GaussianHMM(4, init_params="")
        m1.n_features = 4
        m1.startprob_ = np.array([1/4., 1/4., 1/4., 1/4.])
        m1.transmat_ = np.array(
            [
                [0.2, 0.2, 0.3, 0.3],
                [0.3, 0.2, 0.2, 0.3],
                [0.2, 0.3, 0.3, 0.2],
                [0.3, 0.3, 0.2, 0.2]
            ]
        )
        m1.means_ = np.array([[-1.5], [0], [1.5], [3.]])
        m1.covars_  = np.sqrt([[0.25], [0.25], [0.25], [0.25]])
        return m1

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_mcgrory_titterington(self, implementation):

        random_state = check_random_state(234234)

        model = self.get_mcgrory_titterington()
        sequences = model.sample(500, random_state=random_state)[0]
        lengths = [500]
        model = vhmm.VariationalGaussianHMM(5, n_iter=1000,
                                            implementation=implementation,
                                            tol=1e-9,
                                            random_state=random_state,
                                            verbose=True)
        model.fit(sequences, lengths)
        print(model.monitor_.history)
        print(model.startprob_posterior_)
        print(model.transmat_posterior_)
        print(model.means_posterior_)
        print(model.covars_posterior_)

        em_hmm = hmm.GaussianHMM(n_components=model.n_components,
                                 implementation=implementation,
                                 covariance_type="full"
                                )
        em_hmm.startprob_ = model.startprob_normalized_
        em_hmm.transmat_ = model.transmat_normalized_
        em_hmm.means_ = model.means_posterior_
        em_hmm.covars_ = model.covars_posterior_

        compare_variational_and_em_models(model, em_hmm, sequences, lengths)

