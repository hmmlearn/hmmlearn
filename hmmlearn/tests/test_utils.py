import numpy as np

from hmmlearn.utils import normalize


def test_normalize():
    A = np.random.normal(42., size=128)
    A[np.random.choice(len(A), size=16)] = 0.0
    assert (A == 0.0).any()
    normalize(A)
    assert np.allclose(A.sum(), 1.)


def test_normalize_along_axis():
    A = np.random.normal(42., size=(128, 4))
    for axis in range(A.ndim):
        A[np.random.choice(len(A), size=16), axis] = 0.0
        assert (A[:, axis] == 0.0).any()
        normalize(A, axis=axis)
        assert np.allclose(A.sum(axis=axis), 1.)
