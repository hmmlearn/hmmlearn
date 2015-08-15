import numpy as np
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from hmmlearn.utils import normalize, logsumexp

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


def test_logsumexp():
    x = np.abs(np.random.normal(1e-42, size=100000))
    logx = np.log(x)
    assert_almost_equal(np.exp(logsumexp(logx)), x.sum())

    X = np.vstack([x, x])
    logX = np.vstack([logx, logx])
    assert_array_almost_equal(np.exp(logsumexp(logX, axis=0)), X.sum(axis=0))
    assert_array_almost_equal(np.exp(logsumexp(logX, axis=1)), X.sum(axis=1))
