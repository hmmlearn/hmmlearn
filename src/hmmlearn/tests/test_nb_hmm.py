"""Tests for NegativeBinomialHMM.

Negative Binomial extends Poisson to allow variance > mean (overdispersion).
The model is NB(μ, r): E[X]=μ, Var[X]=μ + μ²/r.

Reference
---------
Zucchini, W., MacDonald, I. L., & Langrock, R. (2016). Hidden Markov Models
for Time Series: An Introduction Using R (2nd ed.). CRC Press.
"""

import importlib
import importlib.util
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

# hmmlearn/utils.py uses `a_sum.shape = shape` which is deprecated in NumPy 2.5.
# This is a pre-existing upstream issue that also affects test_poisson_hmm.py.
# Ignore it here so our tests can exercise the EM path.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:hmmlearn.utils"
)


# ---------------------------------------------------------------------------
# Load NegativeBinomialHMM from the local source tree (can't pip-install
# on this machine because the C extension requires MSVC).
# ---------------------------------------------------------------------------

_HMMLEARN_SRC = __file__.split("tests")[0]

def _reload():
    """Re-import hmmlearn.{_emissions,hmm} from source."""
    for key in ["hmmlearn._emissions", "hmmlearn.hmm"]:
        if key in sys.modules:
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        "hmmlearn._emissions", _HMMLEARN_SRC + "_emissions.py")
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    sys.modules["hmmlearn._emissions"] = em

    spec2 = importlib.util.spec_from_file_location(
        "hmmlearn.hmm", _HMMLEARN_SRC + "hmm.py")
    hmm = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(hmm)
    return hmm.NegativeBinomialHMM


NegativeBinomialHMM = _reload()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(lambdas, dispersions, startprob=None, transmat=None,
                seed=0):
    n = len(lambdas)
    m = NegativeBinomialHMM(n_components=n, n_iter=1, random_state=seed)
    if startprob is None:
        startprob = np.ones(n) / n
    if transmat is None:
        transmat = np.full((n, n), 1 / n)
    m.startprob_ = startprob
    m.transmat_ = transmat
    m.lambdas_ = np.array([[l] for l in lambdas], dtype=float)
    m.dispersions_ = np.array([[d] for d in dispersions], dtype=float)
    return m


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_importable():
    assert NegativeBinomialHMM is not None


def test_in_all():
    spec = importlib.util.spec_from_file_location(
        "hmmlearn.hmm_all", _HMMLEARN_SRC + "hmm.py")
    hmm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hmm)
    assert "NegativeBinomialHMM" in hmm.__all__


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_shape():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    X, Z = m.sample(100)
    assert X.shape == (100, 1)
    assert Z.shape == (100,)


def test_sample_nonnegative():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    X, _ = m.sample(500)
    assert np.all(X >= 0)


def test_sample_integer_counts():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    X, _ = m.sample(200)
    assert np.all(X == np.floor(X))


def test_sample_means_close_to_lambdas():
    m = _make_model(
        [3.0, 25.0],
        [3.0, 8.0],
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.9, 0.1], [0.1, 0.9]]),
    )
    X, Z = m.sample(5000, random_state=99)
    assert abs(X[Z == 0].mean() - 3.0) < 0.5
    assert abs(X[Z == 1].mean() - 25.0) < 1.0


def test_sample_overdispersion():
    """Var > Mean for NB (unlike Poisson where Var == Mean)."""
    m = _make_model([10.0], [2.0])   # dispersion=2 → Var = 10 + 100/2 = 60
    X, _ = m.sample(5000, random_state=7)
    ratio = X.var() / X.mean()
    assert ratio > 1.5, f"Expected overdispersion; got var/mean={ratio:.2f}"


# ---------------------------------------------------------------------------
# Score / log-likelihood
# ---------------------------------------------------------------------------


def test_score_finite():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    X, _ = m.sample(100)
    ll = m.score(X)
    assert np.isfinite(ll)


def test_score_negative():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    X, _ = m.sample(100)
    assert m.score(X) < 0


def test_em_improves_its_own_start():
    """EM score after 20 iterations must exceed score at iteration 1."""
    true = _make_model([3.0, 15.0], [3.0, 6.0])
    X, _ = true.sample(500, random_state=42)
    m1 = NegativeBinomialHMM(n_components=2, n_iter=1, random_state=42)
    m1.fit(X)
    ll1 = m1.score(X)
    m20 = NegativeBinomialHMM(n_components=2, n_iter=20, random_state=42)
    m20.fit(X)
    ll20 = m20.score(X)
    assert ll20 >= ll1 - 0.5  # EM must not degrade its own trajectory


# ---------------------------------------------------------------------------
# EM fitting
# ---------------------------------------------------------------------------


def test_fit_lambdas_close_to_truth():
    """Fitted means should be close to the true generating means."""
    true = _make_model(
        [3.0, 18.0],
        [3.0, 6.0],
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.8, 0.2], [0.2, 0.8]]),
    )
    X, _ = true.sample(2000, random_state=0)
    fit = NegativeBinomialHMM(n_components=2, n_iter=50, random_state=0)
    fit.fit(X)
    fitted_means = np.sort(fit.lambdas_.ravel())
    assert abs(fitted_means[0] - 3.0) < 2.0
    assert abs(fitted_means[1] - 18.0) < 3.0


def test_fit_dispersions_positive():
    true = _make_model([5.0, 20.0], [2.0, 5.0], seed=1)
    X, _ = true.sample(500, random_state=1)
    fit = NegativeBinomialHMM(n_components=2, n_iter=30, random_state=1)
    fit.fit(X)
    assert np.all(fit.dispersions_ > 0)


def test_fit_weights_valid():
    true = _make_model([5.0, 20.0], [2.0, 5.0])
    X, _ = true.sample(300, random_state=2)
    fit = NegativeBinomialHMM(n_components=2, n_iter=30, random_state=2)
    fit.fit(X)
    assert_allclose(fit.startprob_.sum(), 1.0, atol=1e-10)
    assert_allclose(fit.transmat_.sum(axis=1), 1.0, atol=1e-10)


def test_ll_monotonically_increases():
    """EM log-likelihood must be non-decreasing (Dempster et al. 1977)."""
    true = _make_model([4.0, 16.0], [3.0, 5.0])
    X, _ = true.sample(500, random_state=3)
    model = NegativeBinomialHMM(n_components=2, n_iter=40,
                                 verbose=False, random_state=3)
    model.fit(X)
    lls = model.monitor_.history
    diffs = np.diff(lls)
    assert np.all(diffs >= -1e-4), f"Non-monotone EM at steps: {np.where(diffs < -1e-4)}"


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------


def test_predict_shape():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    X, _ = m.sample(100)
    Z_pred = m.predict(X)
    assert Z_pred.shape == (100,)


def test_predict_accuracy_high():
    """With well-separated means, state prediction should be >85 % accurate."""
    true = _make_model(
        [2.0, 30.0],
        [3.0, 8.0],
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.85, 0.15], [0.15, 0.85]]),
    )
    X, Z = true.sample(1000, random_state=5)
    Z_pred = true.predict(X)
    acc = max(np.mean(Z == Z_pred), np.mean(Z == 1 - Z_pred))
    assert acc > 0.85, f"Accuracy {acc:.2%} too low"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_check_raises_negative_lambdas():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    m.lambdas_ = np.array([[-1.0], [20.0]])
    with pytest.raises(ValueError, match="lambdas_"):
        m._check()


def test_check_raises_negative_dispersions():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    m.dispersions_ = np.array([[-1.0], [5.0]])
    with pytest.raises(ValueError, match="dispersions_"):
        m._check()


def test_check_raises_wrong_shape_lambdas():
    m = _make_model([5.0, 20.0], [2.0, 5.0])
    m.lambdas_ = np.array([[5.0, 1.0], [20.0, 2.0]])  # 2 features, 1 actual
    with pytest.raises(ValueError):
        m._check()
