import numpy as np
import pytest

from hmmlearn import hmm

from . import log_likelihood_increasing, make_covar_matrix, normalized


class GaussianHMMTestMixin:
    covariance_type = None  # set by subclasses

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prng = prng = np.random.RandomState(10)
        self.n_components = n_components = 3
        self.n_features = n_features = 3
        self.startprob = prng.rand(n_components)
        self.startprob = self.startprob / self.startprob.sum()
        self.transmat = prng.rand(n_components, n_components)
        self.transmat /= np.tile(self.transmat.sum(axis=1)[:, np.newaxis],
                                 (1, n_components))
        self.means = prng.randint(-20, 20, (n_components, n_features))
        self.covars = make_covar_matrix(
            self.covariance_type, n_components, n_features, random_state=prng
        )

    def test_bad_covariance_type(self):
        with pytest.raises(ValueError):
            h = hmm.GaussianHMM(20, covariance_type='badcovariance_type')
            h.means_ = self.means
            h.covars_ = []
            h.startprob_ = self.startprob
            h.transmat_ = self.transmat
            h._check()

    def test_score_samples_and_decode(self):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type,
                            init_params="st")
        h.means_ = self.means
        h.covars_ = self.covars

        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        h.means_ = 20 * h.means_

        gaussidx = np.repeat(np.arange(self.n_components), 5)
        n_samples = len(gaussidx)
        X = self.prng.randn(n_samples, self.n_features) + h.means_[gaussidx]
        h._init(X)
        ll, posteriors = h.score_samples(X)

        assert posteriors.shape == (n_samples, self.n_components)
        assert np.allclose(posteriors.sum(axis=1), np.ones(n_samples))

        viterbi_ll, stateseq = h.decode(X)
        assert np.allclose(stateseq, gaussidx)

    def test_sample(self, n=1000):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        h.means_ = 20 * self.means
        h.covars_ = np.maximum(self.covars, 0.1)

        X, state_sequence = h.sample(n, random_state=self.prng)
        assert X.shape == (n, self.n_features)
        assert len(state_sequence) == n

    def test_fit(self, params='stmc', n_iter=5, **kwargs):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
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
        # assert log_likelihood_increasing(h, X, lengths, n_iter)

    def test_fit_ignored_init_warns(self, caplog):
        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        h.startprob_ = self.startprob
        h.fit(np.random.randn(100, self.n_components))
        assert len(caplog.records) == 1
        assert "will be overwritten" in caplog.records[0].getMessage()

    def test_fit_too_little_data(self, caplog):
        h = hmm.GaussianHMM(
            self.n_components, self.covariance_type, init_params="")
        h.startprob_ = self.startprob
        h.transmat_ = self.transmat
        h.means_ = 20 * self.means
        h.covars_ = np.maximum(self.covars, 0.1)
        h._init(np.random.randn(5, self.n_components))
        assert len(caplog.records) == 1
        assert "degenerate solution" in caplog.records[0].getMessage()

    def test_fit_sequences_of_different_length(self):
        lengths = [3, 4, 5]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        # This shouldn't raise
        # ValueError: setting an array element with a sequence.
        h.fit(X, lengths=lengths)

    def test_fit_with_length_one_signal(self):
        lengths = [10, 8, 1]
        X = self.prng.rand(sum(lengths), self.n_features)

        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
        # This shouldn't raise
        # ValueError: zero-size array to reduction operation maximum which
        #             has no identity
        h.fit(X, lengths=lengths)

    def test_fit_zero_variance(self):
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

        h = hmm.GaussianHMM(3, self.covariance_type)
        h.fit(X)

    def test_fit_with_priors(self, params='stmc', n_iter=5):
        startprob_prior = 10 * self.startprob + 2.0
        transmat_prior = 10 * self.transmat + 2.0
        means_prior = self.means
        means_weight = 2.0
        covars_weight = 2.0
        if self.covariance_type in ('full', 'tied'):
            covars_weight += self.n_features
        covars_prior = self.covars

        h = hmm.GaussianHMM(self.n_components, self.covariance_type)
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

        # Re-initialize the parameters and check that we can converge to the
        # original parameter values.
        h_learn = hmm.GaussianHMM(self.n_components, self.covariance_type,
                                  params=params)
        h_learn.n_iter = 0
        h_learn.fit(X, lengths=lengths)

        assert log_likelihood_increasing(h_learn, X, lengths, n_iter)

        # Make sure we've converged to the right parameters.
        # a) means
        assert np.allclose(sorted(h.means_.tolist()),
                           sorted(h_learn.means_.tolist()),
                           0.01)
        # b) covars are hard to estimate precisely from a relatively small
        #    sample, thus the large threshold
        assert np.allclose(sorted(h._covars_.tolist()),
                           sorted(h_learn._covars_.tolist()),
                           10)


class TestGaussianHMMWithSphericalCovars(GaussianHMMTestMixin):
    covariance_type = 'spherical'

    def test_fit_startprob_and_transmat(self):
        self.test_fit('st')


class TestGaussianHMMWithDiagonalCovars(GaussianHMMTestMixin):
    covariance_type = 'diag'

    def test_covar_is_writeable(self):
        h = hmm.GaussianHMM(n_components=1, covariance_type="diag",
                            init_params="c")
        X = np.random.normal(size=(1000, 5))
        h._init(X)

        # np.diag returns a read-only view of the array in NumPy 1.9.X.
        # Make sure this doesn't prevent us from fitting an HMM with
        # diagonal covariance matrix. See PR#44 on GitHub for details
        # and discussion.
        assert h._covars_.flags["WRITEABLE"]

    def test_fit_left_right(self):
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
                            params="mct", init_params="cm")
        h.startprob_ = startprob.copy()
        h.transmat_ = transmat.copy()
        h.fit(X)

        assert (h.startprob_[startprob == 0.0] == 0.0).all()
        assert (h.transmat_[transmat == 0.0] == 0.0).all()

        posteriors = h.predict_proba(X)
        assert not np.isnan(posteriors).any()
        assert np.allclose(posteriors.sum(axis=1), 1.)

        score, state_sequence = h.decode(X, algorithm="viterbi")
        assert np.isfinite(score)


class TestGaussianHMMWithTiedCovars(GaussianHMMTestMixin):
    covariance_type = 'tied'


class TestGaussianHMMWithFullCovars(GaussianHMMTestMixin):
    covariance_type = 'full'
