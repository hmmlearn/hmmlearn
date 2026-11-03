import numpy as np
from numpy.testing import assert_allclose

from .. import hmm
from ..utils import fill_covars


def _fit(n_features=2, n_components=2, n_iter=5):
    rng = np.random.RandomState(0)
    X = np.concatenate([rng.normal(0, 1, (300, n_features)),
                        rng.normal(5, 3, (300, n_features))])
    return hmm.GaussianHMM(n_components, "spherical", n_iter=n_iter,
                           random_state=0).fit(X)


def test_spherical_covars_shape_after_init_and_fit():
    h = hmm.GaussianHMM(2, "spherical", random_state=0)
    h._init(np.random.RandomState(0).normal(size=(100, 2)))
    assert h.covars_.shape == (2, 2, 2)
    h = _fit(n_iter=5)
    assert h._covars_.shape == (2,)
    assert h.covars_.shape == (2, 2, 2)


def test_spherical_covars_are_scaled_identities():
    h = _fit(n_features=3, n_components=3)
    for k, cov in enumerate(h.covars_):
        assert_allclose(cov, h._covars_[k] * np.eye(3))


def test_fill_covars_spherical_per_feature_layout():
    # One value per state repeated over features, the layout produced at
    # initialisation, gives one matrix per state.
    covars = np.array([[1., 1.], [4., 4.], [9., 9.]])
    expected = fill_covars(np.array([1., 4., 9.]), "spherical", 3, 2)
    assert_allclose(fill_covars(covars, "spherical", 3, 2), expected)


def test_spherical_samples_use_each_state_variance():
    h = _fit()
    variances = [np.mean(h._covars_[k]) for k in range(h.n_components)]
    assert max(variances) > 4 * min(variances)
    rng = np.random.RandomState(0)
    for k, var in enumerate(variances):
        samples = np.array([h._generate_sample_from_state(k, rng)
                            for _ in range(2000)])
        assert_allclose(samples.var(axis=0), var, rtol=0.15)
