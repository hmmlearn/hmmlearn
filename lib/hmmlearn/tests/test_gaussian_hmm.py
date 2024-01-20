import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.utils import check_random_state

from .. import hmm
from . import assert_log_likelihood_increasing, make_covar_matrix, normalized


class GaussianHMMTestMixin:
    covariance_type = None  # set by subclasses

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prng = prng = np.random.RandomState(10)
        self.n_components = n_components = 3
        self.n_features = n_features = 3
        self.startprob = normalized(prng.rand(n_components))
        self.transmat = normalized(
            prng.rand(n_components, n_components), axis=1)
        self.means = prng.randint(-20, 20, (n_components, n_features))
        self.covars = make_covar_matrix(
            self.covariance_type, n_components, n_features, random_state=prng)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_bad_covariance_type(self, implementation):
        with pytest.raises(ValueError):
            h = hmm.GaussianHMM(20, implementation=implementation,
                                covariance_type='badcovariance_type')
            h.means_ = self.means
            h.covars_ = []
            h.startprob_ = self.startprob
            h.transmat_ = self.transmat
            h._check()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_score_samples_and_decode(self, implementation):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            init_params="st", implementation=implementation)
        h.means_ = self.means
        h.covars_ = self.covars

        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        h.means_ = 20 * h.means_

        gaussidx = np.repeat(np.arange(self.n_components), 5)
        n_samples = len(gaussidx)
        X = (self.prng.randn(n_samples, self.n_features)
             + h.means_[gaussidx])
        h._init(X, [n_samples])
        ll, posteriors = h.score_samples(X)
        assert posteriors.shape == (n_samples, self.n_components)
        assert_allclose(posteriors.sum(axis=1), np.ones(n_samples))

        viterbi_ll, stateseq = h.decode(X)
        assert_allclose(stateseq, gaussidx)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_sample(self, implementation, n=1000):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            implementation=implementation)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        h.means_ = 20 * self.means
        h.covars_ = np.maximum(self.covars, 0.1)

        X, state_sequence = h.sample(n, random_state=self.prng)
        assert X.shape == (n, self.n_features)
        assert len(state_sequence) == n

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit(self, implementation, params='stmc', n_iter=5, **kwargs):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            implementation=implementation)
        h.startprob_ = self.startprob
        h.transmat_ = normalized(
            self.transmat + np.diag(self.prng.rand(self.n_components)), 1)
        h.means_ = 20 * self.means
        h.covars_ = self.covars

        lengths = [10] * 10
        X, _state_sequence = h.sample(sum(lengths), random_state=self.prng)

        # Mess up the parameters and see if we can re-learn them.
        # TODO: change the params and uncomment the check
        h.fit(X, lengths=lengths)
        # assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_criterion(self, implementation):
        random_state = check_random_state(42)
        m1 = hmm.GaussianHMM(self.n_components, init_params="",
            covariance_type=self.covariance_type)
        m1.startprob_ = self.startprob
        m1.transmat_ = self.transmat
        m1.means_ = self.means * 10
        m1.covars_ = self.covars

        X, _ = m1.sample(2000, random_state=random_state)

        aic = []
        bic = []
        ns = [2, 3, 4]
        for n in ns:
            h = hmm.GaussianHMM(n, self.covariance_type, n_iter=500,
                random_state=random_state, implementation=implementation)
            h.fit(X)
            aic.append(h.aic(X))
            bic.append(h.bic(X))

        assert np.all(aic) > 0
        assert np.all(bic) > 0
        # AIC / BIC pick the right model occasionally
        # assert ns[np.argmin(aic)] == self.n_components
        # assert ns[np.argmin(bic)] == self.n_components

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_ignored_init_warns(self, implementation, caplog):
        # This test occasionally will be flaky in learning the model.
        # What is important here, is that the expected log message is produced
        # We can test convergence properties elsewhere.
        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            implementation=implementation)
        h.startprob_ = self.startprob
        h.fit(self.prng.randn(100, self.n_components))
        found = False
        for record in caplog.records:
            if "will be overwritten" in record.getMessage():
                found = True
        assert found, "Did not find expected warning message"

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_too_little_data(self, implementation, caplog):
        h = hmm.GaussianHMM(
            self.n_components, self.covariance_type, init_params="",
            implementation=implementation)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        h.means_ = 20 * self.means
        h.covars_ = np.maximum(self.covars, 0.1)
        h._init(self.prng.randn(5, self.n_components), 5)
        assert len(caplog.records) == 1
        assert "degenerate solution" in caplog.records[0].getMessage()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_sequences_of_different_length(self, implementation):
        lengths = [3, 4, 5]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            implementation=implementation)
        # This shouldn't raise
        # ValueError: setting an array element with a sequence.
        h.fit(X, lengths=lengths)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_with_length_one_signal(self, implementation):
        lengths = [10, 8, 1]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            implementation=implementation)
        # This shouldn't raise
        # ValueError: zero-size array to reduction operation maximum which
        #             has no identity
        h.fit(X, lengths=lengths)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_zero_variance(self, implementation):
        # Example from issue #2 on GitHub.
        X = np.asarray([
            [7.15000000e+02, 5.85000000e+02, 0.00000000e+00, 0.00000000e+00],
            [7.15000000e+02, 5.20000000e+02, 1.04705811e+00, -6.03696289e+01],
            [7.15000000e+02, 4.55000000e+02, 7.20886230e-01, -5.27055664e+01],
            [7.15000000e+02, 3.90000000e+02, -4.57946777e-01, -7.80605469e+01],
            [7.15000000e+02, 3.25000000e+02, -6.43127441e+00, -5.59954834e+01],
            [7.15000000e+02, 2.60000000e+02, -2.90063477e+00, -7.80220947e+01],
            [7.15000000e+02, 1.95000000e+02, 8.45532227e+00, -7.03294373e+01],
            [7.15000000e+02, 1.30000000e+02, 4.09387207e+00, -5.83621216e+01],
            [7.15000000e+02, 6.50000000e+01, -1.21667480e+00, -4.48131409e+01]
        ])
        h = hmm.GaussianHMM(3, self.covariance_type,
                            implementation=implementation)
        h.fit(X)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_with_priors(self, implementation, init_params='mc',
                             params='stmc', n_iter=20):
        # We have a few options to make this a robust test, such as
        # a. increase the amount of training data to ensure convergence
        # b. Only learn some of the parameters (simplify the problem)
        # c. Increase the number of iterations
        #
        # (c) seems to not affect the ci/cd time too much.
        startprob_prior = 10 * self.startprob + 2.0
        transmat_prior = 10 * self.transmat + 2.0
        means_prior = self.means
        means_weight = 2.0
        covars_weight = 2.0
        if self.covariance_type in ('full', 'tied'):
            covars_weight += self.n_features
        covars_prior = self.covars
        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            implementation=implementation)
        h.startprob_ = self.startprob
        h.startprob_prior = startprob_prior
        h.transmat_ = normalized(
            self.transmat + np.diag(self.prng.rand(self.n_components)), 1)
        h.transmat_prior = transmat_prior
        h.means_ = 20 * self.means
        h.means_prior = means_prior
        h.means_weight = means_weight
        h.covars_ = self.covars
        h.covars_prior = covars_prior
        h.covars_weight = covars_weight

        lengths = [200] * 10
        X, _state_sequence = h.sample(sum(lengths), random_state=self.prng)

        # Re-initialize the parameters and check that we can converge to
        # the original parameter values.
        h_learn = hmm.GaussianHMM(self.n_components, self.covariance_type,
                                  init_params=init_params, params=params,
                                  implementation=implementation,)
        # don't use random parameters for testing
        init = 1. / h_learn.n_components
        h_learn.startprob_ = np.full(h_learn.n_components, init)
        h_learn.transmat_ = \
            np.full((h_learn.n_components, h_learn.n_components), init)

        h_learn.n_iter = 0
        h_learn.fit(X, lengths=lengths)

        assert_log_likelihood_increasing(h_learn, X, lengths, n_iter)

        # Make sure we've converged to the right parameters.
        # In general, to account for state switching,
        # compare sorted values.
        # a) means
        assert_allclose(sorted(h.means_.ravel().tolist()),
                        sorted(h_learn.means_.ravel().tolist()),
                        0.01)
        # b) covars are hard to estimate precisely from a relatively small
        #    sample, thus the large threshold

        # account for how we store the covars_compressed
        orig = np.broadcast_to(h._covars_, h_learn._covars_.shape)
        assert_allclose(
            sorted(orig.ravel().tolist()),
            sorted(h_learn._covars_.ravel().tolist()),
            10)


class TestGaussianHMMWithSphericalCovars(GaussianHMMTestMixin):
    covariance_type = 'spherical'

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_issue_385(self, implementation):
        model = hmm.GaussianHMM(n_components=2, covariance_type="spherical")
        model.startprob_ = np.array([0.6, 0.4])
        model.transmat_ = np.array([[0.4, 0.6],
                                    [0.9, 0.1]])
        model.means_ = np.array([[3.0], [5.0]])
        model.covars_ = np.array([[[[4.0]]], [[[3.0]]]])

        # If setting up an HMM to immediately sample from, the easiest thing is
        # to just set n_features.  We could infer it from self.means_ perhaps.
        model.n_features = 1
        covars = model.covars_
        # Make sure covariance is of correct format - the spherical case would
        # throw an exception here.
        model.sample(1000)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_startprob_and_transmat(self, implementation):
        self.test_fit(implementation, 'st')

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_underflow_from_scaling(self, implementation):
        # Setup an ill-conditioned dataset
        data1 = self.prng.normal(0, 1, 100).tolist()
        data2 = self.prng.normal(5, 1, 100).tolist()
        data3 = self.prng.normal(0, 1, 100).tolist()
        data4 = self.prng.normal(5, 1, 100).tolist()
        data = np.concatenate([data1, data2, data3, data4])
        # Insert an outlier
        data[40] = 10000
        data2d = data[:, None]
        lengths = [len(data2d)]
        h = hmm.GaussianHMM(2, n_iter=100, verbose=True,
                            covariance_type=self.covariance_type,
                            implementation=implementation, init_params="")
        h.startprob_ = [0.0, 1]
        h.transmat_ = [[0.4, 0.6], [0.6, 0.4]]
        h.means_ = [[0], [5]]
        h.covars_ = [[1], [1]]
        if implementation == "scaling":
            with pytest.raises(ValueError):
                h.fit(data2d, lengths)

        else:
            h.fit(data2d, lengths)


class TestGaussianHMMWithDiagonalCovars(GaussianHMMTestMixin):
    covariance_type = 'diag'

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_covar_is_writeable(self, implementation):
        h = hmm.GaussianHMM(n_components=1, covariance_type="diag",
                            init_params="c", implementation=implementation)
        X = self.prng.normal(size=(1000, 5))
        h._init(X, 1000)

        # np.diag returns a read-only view of the array in NumPy 1.9.X.
        # Make sure this doesn't prevent us from fitting an HMM with
        # diagonal covariance matrix. See PR#44 on GitHub for details
        # and discussion.
        assert h._covars_.flags["WRITEABLE"]

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_left_right(self, implementation):
        transmat = np.zeros((self.n_components, self.n_components))

        # Left-to-right: each state is connected to itself and its
        # direct successor.
        for i in range(self.n_components):
            if i == self.n_components - 1:
                transmat[i, i] = 1.0
            else:
                transmat[i, i] = transmat[i, i + 1] = 0.5

        # Always start in first state
        startprob = np.zeros(self.n_components)
        startprob[0] = 1.0

        lengths = [10, 8, 1]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GaussianHMM(self.n_components, covariance_type="diag",
                            params="mct", init_params="cm",
                            implementation=implementation)
        h.startprob_ = startprob.copy()
        h.transmat_ = transmat.copy()
        h.fit(X)

        assert (h.startprob_[startprob == 0.0] == 0.0).all()
        assert (h.transmat_[transmat == 0.0] == 0.0).all()

        posteriors = h.predict_proba(X)
        assert not np.isnan(posteriors).any()
        assert_allclose(posteriors.sum(axis=1), 1.)

        score, state_sequence = h.decode(X, algorithm="viterbi")
        assert np.isfinite(score)


class TestGaussianHMMWithTiedCovars(GaussianHMMTestMixin):
    covariance_type = 'tied'


class TestGaussianHMMWithFullCovars(GaussianHMMTestMixin):
    covariance_type = 'full'
