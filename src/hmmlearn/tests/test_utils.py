import numpy as np
import pytest
from numpy.testing import assert_allclose

from hmmlearn.utils import normalize, fill_covars


def test_normalize():
    A = np.random.normal(42., size=128)
    A[np.random.choice(len(A), size=16)] = 0.0
    assert (A == 0.0).any()
    normalize(A)
    assert_allclose(A.sum(), 1.)


def test_normalize_along_axis():
    A = np.random.normal(42., size=(128, 4))
    for axis in range(A.ndim):
        A[np.random.choice(len(A), size=16), axis] = 0.0
        assert (A[:, axis] == 0.0).any()
        normalize(A, axis=axis)
        assert_allclose(A.sum(axis=axis), 1.)


@pytest.mark.parametrize("shape, axis", [
    ((2, 3), 0), ((2, 3), 1), ((2, 3), -2), ((2, 3), -1),
    ((2, 3, 4), 0), ((2, 3, 4), 1), ((2, 3, 4), 2),
    ((2, 3, 4), -3), ((2, 3, 4), -2), ((2, 3, 4), -1),
])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalize_zero_sums(shape, axis, dtype):
    a = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
    zero_slice = [slice(None)] * a.ndim
    zero_slice[(axis + 1) % a.ndim] = 0
    a[tuple(zero_slice)] = 0
    zero_mask = a == 0
    nonzero_sums = a.sum(axis=axis) != 0

    with np.errstate(divide="raise", invalid="raise"):
        assert normalize(a, axis=axis) is None

    assert a.shape == shape
    assert np.isfinite(a).all()
    assert_allclose(a[zero_mask], 0)
    assert_allclose(a.sum(axis=axis)[nonzero_sums], 1, atol=1e-7)
    assert_allclose(a.sum(axis=axis)[~nonzero_sums], 0)


@pytest.mark.parametrize("axis", [0, 1, -2, -1])
def test_normalize_all_zero(axis):
    a = np.zeros((2, 3))

    with np.errstate(divide="raise", invalid="raise"):
        assert normalize(a, axis=axis) is None

    assert_allclose(a, np.zeros((2, 3)))


def test_fill_covars():
    full = np.arange(12).reshape(3, 2, 2) + 1
    np.testing.assert_equal(fill_covars(full, 'full', 3, 2), full)

    diag = np.arange(6).reshape(3, 2) + 1
    expected = np.array([[[1, 0], [0, 2]],
                         [[3, 0], [0, 4]],
                         [[5, 0], [0, 6]]])
    np.testing.assert_equal(fill_covars(diag, 'diag', 3, 2), expected)

    tied = np.arange(4).reshape(2, 2) + 1
    expected = np.array([[[1, 2], [3, 4]],
                         [[1, 2], [3, 4]],
                         [[1, 2], [3, 4]]])
    np.testing.assert_equal(fill_covars(tied, 'tied', 3, 2), expected)

    spherical = np.array([1, 2, 3])
    expected = np.array([[[1, 0], [0, 1]],
                         [[2, 0], [0, 2]],
                         [[3, 0], [0, 3]]])
    np.testing.assert_equal(
        fill_covars(spherical, 'spherical', 3, 2), expected)
