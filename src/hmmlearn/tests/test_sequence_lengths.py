import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from hmmlearn.hmm import CategoricalHMM
from hmmlearn import _hmmc, _utils


def new_hmm(implementation):
    h = CategoricalHMM(1, n_features=2, init_params="", n_iter=1,
                       implementation=implementation)
    h.startprob_ = np.ones(1)
    h.transmat_ = np.ones((1, 1))
    h.emissionprob_ = np.array([[0.5, 0.5]])
    return h


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("method", [
    "fit", "score", "score_samples", "decode", "predict", "predict_proba",
])
@pytest.mark.parametrize("lengths", [
    [], [0, 2], [1, 0, 1], [2, 0], [-1, 3], [3, -1],
    [0.5, 1.5], [1., 1.], [[1, 1]], [True, True],
])
def test_invalid_sequence_lengths(implementation, method, lengths):
    h = new_hmm(implementation)
    with pytest.raises(ValueError, match="positive integers"):
        getattr(h, method)(np.array([[0], [1]]), lengths)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("lengths", [
    None, [2], [1, 1], (1, 1), np.array([1, 1], dtype=np.uint8),
])
def test_valid_sequence_lengths(implementation, lengths):
    h = new_hmm(implementation)
    X = np.array([[0], [1]])
    assert_allclose(h.score(X, lengths), 2 * np.log(0.5))
    log_prob, posteriors = h.score_samples(X, lengths)
    assert_allclose(log_prob, 2 * np.log(0.5))
    assert_array_equal(posteriors, np.ones((2, 1)))
    assert_array_equal(h.predict(X, lengths), [0, 0])
    h.fit(X, lengths)
    assert_allclose(h.emissionprob_, [[0.5, 0.5]])


def test_sequence_lengths_sum_mismatch():
    h = new_hmm("log")
    with pytest.raises(ValueError, match="doesn't sum to 2 samples"):
        h.score(np.array([[0], [1]]), [1])


@pytest.mark.parametrize("lengths", [
    np.array([2**64 - 1, 3], dtype=np.uint64),
    np.array([2**63 - 1, 2**63 - 1, 4], dtype=np.int64),
    [2**63 - 1, 2**63 - 1, 4],
])
def test_sequence_lengths_overflow(lengths):
    with pytest.raises(ValueError, match="doesn't sum to 2 samples"):
        _utils.split_X_lengths(np.array([[0], [1]]), lengths)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("method", [
    "fit", "score", "score_samples", "decode", "predict", "predict_proba",
    "aic", "bic",
])
def test_overflow_lengths_rejected_before_inference(
        monkeypatch, implementation, method):
    def unexpected_native_call(*args):
        pytest.fail("invalid lengths reached native inference")

    for name in ("forward_log", "forward_scaling", "viterbi"):
        monkeypatch.setattr(_hmmc, name, unexpected_native_call)
    h = new_hmm(implementation)
    lengths = np.array([2**64 - 1, 3], dtype=np.uint64)
    with pytest.raises(ValueError, match="doesn't sum to 2 samples"):
        getattr(h, method)(np.array([[0], [1]]), lengths)
