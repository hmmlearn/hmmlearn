import numpy as np
from nose.tools import assert_almost_equal

from hmmlearn.utils import normalize

rng = np.random.RandomState(0)
np.seterr(all='warn')


def test_normalize():
    A = np.random.normal((128, 4), 42.)
    normalize(A)
    assert_almost_equal(A.sum(), 1.)


def test_normalize_along_axis():
    A = np.random.normal((128, 4), 42.)
    for axis in range(A.ndim):
        normalize(A, axis=axis)
        assert_almost_equal(A.sum(axis=axis), 1.)
