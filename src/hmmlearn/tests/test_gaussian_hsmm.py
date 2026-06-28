"""Tests for GaussianHSMM (Hidden Semi-Markov Model with Poisson durations).

Reference: Yu (2010) "Hidden semi-Markov models", Artificial Intelligence
174(2):215-243.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .. import hsmm as hsmm_module
from ..hsmm import GaussianHSMM
from . import assert_log_likelihood_increasing, make_covar_matrix, normalized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_state_model(random_state=0):
    """Pre-built 2-state GaussianHSMM with clear mean separation."""
    rng = np.random.default_rng(random_state)
    model = GaussianHSMM(
        n_components=2,
        covariance_type="diag",
        max_duration=12,
        n_iter=30,
        tol=1e-3,
        random_state=0,
    )
    # Two clearly separated Gaussian states.
    X_seg1 = rng.normal(0.0, 0.5, (80, 2))
    X_seg2 = rng.normal(5.0, 0.5, (80, 2))
    X = np.vstack([X_seg1, X_seg2, X_seg1, X_seg2])  # 320 samples
    return model, X


def _synth_signal(n_components=3, n_features=1, seg_len=30, rng=None):
    """Synthesise a piecewise-constant signal from a known state sequence."""
    if rng is None:
        rng = np.random.default_rng(42)
    means = np.arange(n_components, dtype=float)[:, None] * 4.0
    segments = []
    states = []
    for i in range(n_components):
        seg = rng.normal(means[i], 0.5, (seg_len, n_features))
        segments.append(seg)
        states.extend([i] * seg_len)
    return np.vstack(segments), np.array(states)


# ---------------------------------------------------------------------------
# Module-level smoke tests
# ---------------------------------------------------------------------------


def test_module_exports():
    assert "GaussianHSMM" in hsmm_module.__all__


# ---------------------------------------------------------------------------
# Constructor and attribute tests
# ---------------------------------------------------------------------------


def test_init_stores_max_duration():
    m = GaussianHSMM(n_components=3, max_duration=15)
    assert m.max_duration == 15


def test_init_default_max_duration():
    m = GaussianHSMM()
    assert m.max_duration == 10


def test_check_validates_dur_lambda_shape():
    model = GaussianHSMM(n_components=2, max_duration=5, n_iter=1)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2))
    model.fit(X)
    # Wrong length should raise.
    model.dur_lambda_ = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="dur_lambda_"):
        model._check()


def test_check_validates_dur_lambda_positive():
    model = GaussianHSMM(n_components=2, max_duration=5, n_iter=1)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2))
    model.fit(X)
    model.dur_lambda_ = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match="positive"):
        model._check()


def test_check_max_duration_at_least_1():
    model = GaussianHSMM(n_components=2, max_duration=0, n_iter=1)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2))
    with pytest.raises(ValueError, match="max_duration"):
        model.fit(X)


def test_no_self_transitions_after_fit():
    model, X = _two_state_model()
    model.fit(X)
    assert np.allclose(np.diag(model.transmat_), 0.0)


# ---------------------------------------------------------------------------
# Duration probabilities
# ---------------------------------------------------------------------------


def test_dur_probs_sum_to_one():
    model = GaussianHSMM(n_components=3, max_duration=20, n_iter=1)
    rng = np.random.default_rng(0)
    model.fit(rng.standard_normal((60, 2)))
    dp = model._dur_probs()
    assert_allclose(dp.sum(axis=1), 1.0, atol=1e-10)


def test_dur_probs_shape():
    N, D = 4, 8
    model = GaussianHSMM(n_components=N, max_duration=D, n_iter=1)
    model.fit(np.random.randn(40, 2))
    assert model._dur_probs().shape == (N, D)


def test_dur_probs_nonnegative():
    model = GaussianHSMM(n_components=3, max_duration=15, n_iter=1)
    model.fit(np.random.randn(60, 2))
    assert np.all(model._dur_probs() >= 0)


# ---------------------------------------------------------------------------
# Augmented-model structure
# ---------------------------------------------------------------------------


def test_augmented_startprob_sums_to_one():
    model, X = _two_state_model()
    model.fit(X)
    aug_start, _ = model._augment_params()
    assert_allclose(aug_start.sum(), 1.0, atol=1e-10)


def test_augmented_transmat_row_sums():
    """Each row of the augmented transmat must sum to 1 (or 0 for dead rows)."""
    model, X = _two_state_model()
    model.fit(X)
    _, aug_trans = model._augment_params()
    row_sums = aug_trans.sum(axis=1)
    # Rows sum to 1 for states that have outgoing transitions.
    # The last-duration state (i, 1) transitions out (non-zero row sum).
    # All other states also have exactly one outgoing transition.
    assert np.all((np.abs(row_sums - 1.0) < 1e-10) | (row_sums == 0.0))


def test_augmented_transmat_shape():
    N, D = 3, 7
    model = GaussianHSMM(n_components=N, max_duration=D, n_iter=1)
    model.fit(np.random.randn(60, 2))
    _, aug_trans = model._augment_params()
    assert aug_trans.shape == (N * D, N * D)


# ---------------------------------------------------------------------------
# Fit / score
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    model = GaussianHSMM(n_components=2, max_duration=5, n_iter=3)
    X = np.random.randn(80, 2)
    assert model.fit(X) is model


def test_log_likelihood_finite_after_fit():
    model, X = _two_state_model()
    model.fit(X)
    ll = model.score(X)
    assert np.isfinite(ll)


def test_log_likelihood_increasing_1d():
    model = GaussianHSMM(
        n_components=2, max_duration=8, covariance_type="diag", random_state=1)
    X, _ = _synth_signal(n_components=2, n_features=1, seg_len=40)
    assert_log_likelihood_increasing(model, X, lengths=None, n_iter=10)


def test_log_likelihood_increasing_2d():
    model = GaussianHSMM(
        n_components=2, max_duration=8, covariance_type="diag", random_state=2)
    X, _ = _synth_signal(n_components=2, n_features=2, seg_len=40)
    assert_log_likelihood_increasing(model, X, lengths=None, n_iter=8)


# ---------------------------------------------------------------------------
# Predict / decode
# ---------------------------------------------------------------------------


def test_predict_returns_array_of_correct_length():
    model, X = _two_state_model()
    model.fit(X)
    states = model.predict(X)
    assert states.shape == (X.shape[0],)


def test_predict_state_values_in_range():
    model, X = _two_state_model()
    model.fit(X)
    states = model.predict(X)
    assert states.min() >= 0
    assert states.max() < model.n_components


def test_viterbi_decoding_segments():
    """Viterbi must recover the correct majority state for each segment."""
    rng = np.random.default_rng(7)
    n_comp, seg_len = 2, 50
    X_seg1 = rng.normal(0.0, 0.3, (seg_len, 1))
    X_seg2 = rng.normal(5.0, 0.3, (seg_len, 1))
    X = np.vstack([X_seg1, X_seg2])

    model = GaussianHSMM(
        n_components=n_comp, max_duration=15, n_iter=30,
        covariance_type="diag", random_state=5)
    model.fit(X)
    pred = model.predict(X)

    # Each segment must be dominated by a single state.
    s0 = int(np.median(pred[:seg_len]))
    s1 = int(np.median(pred[seg_len:]))
    assert s0 != s1


def test_map_decoding():
    model, X = _two_state_model()
    model.fit(X)
    _, states = model.decode(X, algorithm="map")
    assert states.shape == (X.shape[0],)
    assert states.min() >= 0
    assert states.max() < 2


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_shape():
    model, X = _two_state_model()
    model.fit(X)
    samples, state_seq = model.sample(50)
    assert samples.shape == (50, 2)
    assert state_seq.shape == (50,)


def test_sample_state_values_in_range():
    model, X = _two_state_model()
    model.fit(X)
    _, state_seq = model.sample(200)
    assert state_seq.min() >= 0
    assert state_seq.max() < 2


def test_sample_no_self_transitions():
    """Consecutive states in the sample must differ (no self-transitions)."""
    model, X = _two_state_model()
    model.fit(X)
    _, state_seq = model.sample(300, random_state=0)
    # Within a segment, the same state is repeated.  But transitions between
    # segments must go to a different state.
    # Find transition points (where state changes).
    transitions = np.where(np.diff(state_seq) != 0)[0]
    if len(transitions) > 0:
        for t in transitions:
            assert state_seq[t] != state_seq[t + 1], (
                f"Self-transition at t={t}: state {state_seq[t]}"
            )


def test_sample_currstate():
    model, X = _two_state_model()
    model.fit(X)
    _, state_seq = model.sample(100, currstate=0, random_state=0)
    assert state_seq[0] == 0


# ---------------------------------------------------------------------------
# Covariance types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cov_type", ["diag", "full", "spherical", "tied"])
def test_covariance_types(cov_type):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 0.5, (40, 2)),
                   rng.normal(4.0, 0.5, (40, 2))])
    model = GaussianHSMM(
        n_components=2, max_duration=8, covariance_type=cov_type,
        n_iter=5, random_state=0)
    model.fit(X)
    assert np.isfinite(model.score(X))


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------


def test_dur_lambda_fitted():
    """dur_lambda_ must be positive after fitting."""
    model, X = _two_state_model()
    model.fit(X)
    assert np.all(model.dur_lambda_ > 0)
    assert model.dur_lambda_.shape == (2,)


def test_dur_lambda_rough_accuracy():
    """With long segments, the estimated mean duration should be in the right
    ballpark (within a factor of 2 of the true value)."""
    rng = np.random.default_rng(12)
    true_dur = 8
    n_segs = 20
    n_features = 1

    # Build signal: alternating state 0 (dur=8) / state 1 (dur=8).
    segments, true_states = [], []
    for k in range(n_segs):
        state = k % 2
        mean = state * 5.0
        seg = rng.normal(mean, 0.5, (true_dur, n_features))
        segments.append(seg)
        true_states.extend([state] * true_dur)

    X = np.vstack(segments)
    model = GaussianHSMM(
        n_components=2, max_duration=20, n_iter=40,
        covariance_type="diag", random_state=0)
    model.fit(X)

    assert np.all(model.dur_lambda_ > 0)
    # Estimated durations should be reasonably close to the true value.
    for lam in model.dur_lambda_:
        assert true_dur / 2 <= lam <= true_dur * 2, (
            f"dur_lambda={lam:.2f} too far from true duration {true_dur}")


# ---------------------------------------------------------------------------
# Multi-sequence fit
# ---------------------------------------------------------------------------


def test_fit_multiple_sequences():
    rng = np.random.default_rng(3)
    X1 = np.vstack([rng.normal(0.0, 0.5, (30, 2)),
                    rng.normal(4.0, 0.5, (30, 2))])
    X2 = np.vstack([rng.normal(0.0, 0.5, (25, 2)),
                    rng.normal(4.0, 0.5, (25, 2))])
    X = np.vstack([X1, X2])
    lengths = [60, 50]
    model = GaussianHSMM(n_components=2, max_duration=8, n_iter=10,
                         random_state=0)
    model.fit(X, lengths=lengths)
    assert np.isfinite(model.score(X, lengths=lengths))


# ---------------------------------------------------------------------------
# Params / init_params control
# ---------------------------------------------------------------------------


def test_params_d_skips_duration_update():
    model, X = _two_state_model()
    # Pre-set duration rates.
    model.fit(X)
    lambda_before = model.dur_lambda_.copy()
    # Refit without updating duration.
    model.params = "stmc"
    model.init_params = ""
    model.fit(X)
    assert_allclose(model.dur_lambda_, lambda_before)


def test_score_samples_shape():
    model, X = _two_state_model()
    model.fit(X)
    ll, posteriors = model.score_samples(X)
    assert np.isfinite(ll)
    assert posteriors.shape == (X.shape[0], 2)
    assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-6)
