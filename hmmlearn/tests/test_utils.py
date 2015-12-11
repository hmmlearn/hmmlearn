import numpy as np

from hmmlearn.utils import normalize, logsumexp

rng = np.random.RandomState(0)
np.seterr(all='warn')


def test_normalize():
    A = np.random.normal((128, 4), 42.)
    normalize(A)
    assert np.allclose(A.sum(), 1.)


def test_normalize_along_axis():
    A = np.random.normal((128, 4), 42.)
    for axis in range(A.ndim):
        normalize(A, axis=axis)
        assert np.allclose(A.sum(axis=axis), 1.)


def test_logsumexp():
    x = np.abs(np.random.normal(1e-42, size=100000))
    logx = np.log(x)
    assert np.allclose(np.exp(logsumexp(logx)), x.sum())

    X = np.vstack([x, x])
    logX = np.vstack([logx, logx])
    assert np.allclose(np.exp(logsumexp(logX, axis=0)), X.sum(axis=0))
    assert np.allclose(np.exp(logsumexp(logX, axis=1)), X.sum(axis=1))
