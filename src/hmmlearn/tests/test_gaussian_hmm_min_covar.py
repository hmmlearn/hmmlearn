import numpy as np
from numpy.testing import assert_allclose
import pytest

from .. import hmm


def _rounded_sequences(n_sequences=5, length=400, scale=0.5, seed=0):
    # Rounded Gaussian values: about two thirds are exactly zero, so one
    # state can explain the zeros with a vanishing variance.
    rng = np.random.RandomState(seed)
    X = np.round(rng.normal(0, scale, (n_sequences, length)))
    return X.reshape(-1, 1), [length] * n_sequences


@pytest.mark.parametrize("covariance_type", ["spherical", "diag", "full"])
def test_min_covar_is_a_floor_after_fit(covariance_type):
    # Without the floor in the M-step, the variance of the "zeros" state
    # ends near 3e-5, 30 times below min_covar.
    X, lengths = _rounded_sequences()
    h = hmm.GaussianHMM(2, covariance_type, min_covar=1e-3, n_iter=100,
                        tol=1e-4, random_state=0)
    h.fit(X, lengths)
    eig = np.linalg.eigvalsh(h.covars_)
    assert eig.min() >= 1e-3 * (1 - 1e-9)
    assert eig.min() <= 1e-3 * (1 + 1e-9)  # the floor binds


@pytest.mark.parametrize("covariance_type", ["full", "tied"])
def test_min_covar_floors_eigenvalues(covariance_type):
    # The second feature is an exact copy of the first: the covariance is
    # singular without the floor.
    rng = np.random.RandomState(1)
    x = np.round(rng.normal(0, 0.5, 1000))
    X = np.column_stack([x, x])
    h = hmm.GaussianHMM(2, covariance_type, min_covar=1e-3, n_iter=20,
                        random_state=0)
    h.fit(X, [100] * 10)
    assert np.linalg.eigvalsh(h.covars_).min() >= 1e-3 * (1 - 1e-9)


@pytest.mark.parametrize("covariance_type",
                         ["spherical", "diag", "full", "tied"])
def test_min_covar_monotone_em(covariance_type):
    # The floored update is the exact M-step under the constraint, so the
    # log-likelihood still never decreases.
    X, lengths = _rounded_sequences()
    h = hmm.GaussianHMM(2, covariance_type, min_covar=1e-2, n_iter=30,
                        tol=-np.inf, random_state=0)
    h.fit(X, lengths)
    history = np.asarray(h.monitor_.history)
    assert (np.diff(history) >= -1e-8 * np.abs(history[1:])).all()


@pytest.mark.parametrize("covariance_type",
                         ["spherical", "diag", "full", "tied"])
def test_min_covar_inactive_floor_changes_nothing(covariance_type):
    # From the same starting point, a floor that never binds gives exactly
    # the fit without a floor (min_covar=0 disables it).
    rng = np.random.RandomState(2)
    X = np.concatenate([rng.normal(0, 1, (500, 2)),
                        rng.normal(4, 2, (500, 2))])
    start = hmm.GaussianHMM(2, covariance_type, random_state=0)
    start._init(X)
    fits = []
    for min_covar in (1e-3, 0.0):
        h = hmm.GaussianHMM(2, covariance_type, min_covar=min_covar,
                            n_iter=20, init_params="", random_state=0)
        h.startprob_ = start.startprob_
        h.transmat_ = start.transmat_
        h.means_ = start.means_
        h.covars_ = start._covars_
        fits.append(h.fit(X))
    assert np.linalg.eigvalsh(fits[0].covars_).min() > 1e-2
    assert_allclose(fits[0].means_, fits[1].means_, rtol=0, atol=0)
    assert_allclose(fits[0].covars_, fits[1].covars_, rtol=0, atol=0)
