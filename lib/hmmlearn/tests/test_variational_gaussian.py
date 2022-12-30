import numpy as np
import pytest
from sklearn.utils import check_random_state

from hmmlearn import hmm, vhmm
from . import (
    assert_log_likelihood_increasing, compare_variational_and_em_models)


def get_mcgrory_titterington():
    m1 = hmm.GaussianHMM(4, init_params="")
    m1.n_features = 4
    m1.startprob_ = np.array([1/4., 1/4., 1/4., 1/4.])
    m1.transmat_ = np.array([[0.2, 0.2, 0.3, 0.3],
                             [0.3, 0.2, 0.2, 0.3],
                             [0.2, 0.3, 0.3, 0.2],
                             [0.3, 0.3, 0.2, 0.2]])
    m1.means_ = np.array([[-1.5], [0], [1.5], [3.]])
    m1.covars_ = np.sqrt([[0.25], [0.25], [0.25], [0.25]])
    return m1


def get_mcgrory_titterington2d():
    """ A subtle variation on the 1D Case..."""
    m1 = hmm.GaussianHMM(4, init_params="", covariance_type="tied")
    m1.n_features = 4
    m1.startprob_ = np.array([1/4., 1/4., 1/4., 1/4.])
    m1.transmat_ = np.array([[0.2, 0.2, 0.3, 0.3],
                             [0.3, 0.2, 0.2, 0.3],
                             [0.2, 0.3, 0.3, 0.2],
                             [0.3, 0.3, 0.2, 0.2]])
    m1.means_ = np.array([[-1.5, -1.5], [0, 0], [1.5, 1.5], [3., 3]])
    m1.covars_ = np.sqrt([[0.25, 0], [0, .25]])
    return m1


def get_sequences(length, N, model, rs=None):
    sequences = []
    lengths = []
    rs = check_random_state(rs)
    for i in range(N):
        sequences.append(
            model.sample(length, random_state=rs)[0])
        lengths.append(len(sequences[-1]))

    sequences = np.concatenate(sequences)
    return sequences, lengths


class _TestGaussian:

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_mcgrory_titterington1d(self, implementation):
        random_state = check_random_state(234234)

        sequences, lengths = get_sequences(500, 1,
                                           model=get_mcgrory_titterington(),
                                           rs=random_state)
        model = vhmm.VariationalGaussianHMM(
            5, n_iter=1000, tol=1e-9, random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation)
        model.fit(sequences, lengths)
        # Perform one check that we are converging to the right answer
        assert (model.means_posterior_[-1][0]
                == pytest.approx(self.test_fit_mcgrory_titterington1d_mean)), \
            model.means_posterior_

        em_hmm = hmm.GaussianHMM(
            n_components=model.n_components,
            implementation=implementation,
            covariance_type=self.covariance_type,
        )
        em_hmm.startprob_ = model.startprob_
        em_hmm.transmat_ = model.transmat_
        em_hmm.means_ = model.means_posterior_
        em_hmm.covars_ = model._covars_

        compare_variational_and_em_models(model, em_hmm, sequences, lengths)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_mcgrory_titterington2d(self, implementation):
        sequences, lengths = get_sequences(100, 1,
                                           model=get_mcgrory_titterington2d())

        model = vhmm.VariationalGaussianHMM(
            5, n_iter=1000, tol=1e-9, random_state=None,
            covariance_type=self.covariance_type,
            implementation=implementation)
        model.fit(sequences, lengths)

        em_hmm = hmm.GaussianHMM(n_components=model.n_components,
                                 implementation=implementation,
                                 covariance_type=self.covariance_type)
        em_hmm.startprob_ = model.startprob_
        em_hmm.transmat_ = model.transmat_
        em_hmm.means_ = model.means_posterior_
        em_hmm.covars_ = model._covars_

        compare_variational_and_em_models(model, em_hmm, sequences, lengths)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_incorrect_init(self, implementation):
        sequences, lengths = get_sequences(50, 10,
                                           model=get_mcgrory_titterington())

        with pytest.raises(NotImplementedError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9,
                covariance_type="incorrect",
                implementation=implementation)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Manually setup means - should converge
        model = vhmm.VariationalGaussianHMM(
            4, n_iter=500, tol=1e-9, init_params="stc",
            covariance_type=self.covariance_type,
            implementation=implementation)
        model.means_prior_ = [[1], [1], [1], [1]]
        model.means_posterior_ = [[2], [1], [3], [4]]
        model.beta_prior_ = [1, 1, 1, 1]
        model.beta_posterior_ = [1, 1, 1, 1]
        assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Means have wrong shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stc",
                covariance_type=self.covariance_type,
                implementation=implementation)
            model.means_prior_ = [[1], [1], [1]]
            model.means_posterior_ = [[1], [1], [1], [1]]
            model.beta_prior_ = [1, 1, 1, 1]
            model.beta_posterior_ = [1, 1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stc",
                covariance_type=self.covariance_type,
                implementation=implementation)
            model.means_prior_ = [[1], [1], [1], [1]]
            model.means_posterior_ = [[1], [1], [1]]
            model.beta_prior_ = [1, 1, 1, 1]
            model.beta_posterior_ = [1, 1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # beta's have wrong shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stc",
                covariance_type=self.covariance_type,
                implementation=implementation)
            model.means_prior_ = [[1], [1], [1], [1]]
            model.means_posterior_ = [[2], [1], [3], [4]]
            model.beta_prior_ = [1, 1, 1]
            model.beta_posterior_ = [1, 1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stc",
                covariance_type=self.covariance_type,
                implementation=implementation)
            model.means_prior_ = [[1], [1], [1], [1]]
            model.means_posterior_ = [[2], [1], [3], [4]]
            model.beta_prior_ = [1, 1, 1, 1]
            model.beta_posterior_ = [1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)


class TestFull(_TestGaussian):

    covariance_type = "full"
    test_fit_mcgrory_titterington1d_mean = 1.41058519

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_incorrect_init_covariance(self, implementation):
        random_state = check_random_state(234234)
        sequences, lengths = get_sequences(
            50, 10, model=get_mcgrory_titterington())

        # dof's have wrong shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stc",
                covariance_type="full",
                implementation=implementation)
            model.dof_prior_ = [1, 1, 1]
            model.dof_posterior_ = [1, 1, 1, 1]
            model.scale_prior_ = [[[2.]], [[2.]], [[2.]], [[2]]]
            model.scale_posterior_ = [[[2.]], [[2.]], [[2.]], [[2.]]]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Dof posterior is used to setup the W's
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stc",
                covariance_type="full",
                implementation=implementation)
            model.dof_prior_ = [1, 1, 1, 1]
            model.dof_posterior_ = [1, 1, 1]
            model.scale_prior_ = [[[2.]], [[2.]], [[2.]], [[2]]]
            model.scale_posterior_ = [[[2.]], [[2.]], [[2.]], [[2.]]]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Skip the initialization of the Ws, and fail during _check()
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stc",
                covariance_type="full",
                implementation=implementation)
            model.dof_prior_ = [1, 1, 1, 1]
            model.dof_posterior_ = [1, 1, 1]
            model.scale_prior_ = [[[2.]], [[2.]], [[2.]], [[2]]]
            model.scale_posterior_ = [[[2.]], [[2.]], [[2.]], [[2.]]]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)
        # scales's have wrong shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stmbd",
                covariance_type="full",
                implementation=implementation)
            model.dof_prior_ = [1, 1, 1, 1]
            model.dof_posterior_ = [1, 1, 1, 1]
            model.scale_prior_ = [[[2.]], [[2.]], [[2.]]]  # This is wrong
            model.scale_posterior_ = [[[2.]], [[2.]], [[2.]], [[2.]]]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stm",
                covariance_type="full",
                implementation=implementation)
            model.dof_prior_ = [1, 1, 1, 1]
            model.dof_posterior_ = [1, 1, 1, 1]
            model.scale_prior_ = [[[2.]], [[2.]], [[2.]], [[2]]]
            model.scale_posterior_ = [[[2.]], [[2.]], [[2.]]]  # this is wrong
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Manually setup covariance
        with pytest.raises(NotImplementedError):
            model = vhmm.VariationalGaussianHMM(
                4, n_iter=500, tol=1e-9, init_params="stm",
                covariance_type="incorrect",
                implementation=implementation)
            model.dof_prior_ = [1., 1., 1., 1.,]
            model.dof_posterior_ = [1., 1., 1., 1.,]
            model.scale_prior_ = [[[2.]], [[2.]], [[2.]], [[2.]]],
            model.scale_posterior_ = [[[2.]], [[2.]], [[2.]], [[2.]]],
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Set priors via params
        model = vhmm.VariationalGaussianHMM(
            4, n_iter=500, tol=1e-9, random_state=random_state,
            covariance_type="full",
            implementation=implementation,
            means_prior=[[0.], [0.], [0.], [0.]],
            beta_prior=[2., 2., 2., 2.],
            dof_prior=[2., 2., 2., 2.],
            scale_prior=[[[2.]], [[2.]], [[2.]], [[2.]]])
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert np.all(model.means_prior_ == 0)
        assert np.all(model.beta_prior_ == 2.)
        assert np.all(model.dof_prior_ == 2.)
        assert np.all(model.scale_prior_ == 2.)


# As other covariance types are implemented, refactor this.
class NotImplementedYet:

    covariance_type = None

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_mcgrory_titterington(self, implementation):
        random_state = check_random_state(234234)
        sequences, lengths = get_sequences(500, 1,
                                           model=get_mcgrory_titterington())
        model = vhmm.VariationalGaussianHMM(
            5, n_iter=1000, tol=1e-9, random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation)
        with pytest.raises(NotImplementedError):
            model.fit(sequences, lengths)


class TestTied(_TestGaussian):
    test_fit_mcgrory_titterington1d_mean = 1.4774254
    covariance_type = "tied"


class TestSpherical(NotImplementedYet):
    covariance_type = "spherical"


class TestDiagonal(NotImplementedYet):
    covariance_type = "diag"
