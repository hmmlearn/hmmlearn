import numpy as np


def normalize(a, axis=None):
    """Normalize the input array so that it sums to 1.

    Parameters
    ----------
    a: array_like
        Non-normalized input data.
    axis: int
        Dimension along which normalization is performed.

    Returns
    -------
    res: array, shape (n_samples, n_features)
        A with values normalized (summing to 1) along the prescribed axis

    WARNING: Modifies the array inplace.
    """
    a += np.finfo(float).eps

    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum

    # TODO: should return nothing, since the operation is inplace.
    return a


def exp_mask_zero(a):
    """Computes the exponent of input elements masking underflows."""
    with np.errstate(under="ignore"):
        out = np.exp(a)
    out[out == 0] = np.finfo(float).eps
    return out


def log_mask_zero(a):
    """Computes the log of input elements masking underflows."""
    with np.errstate(divide="ignore"):
        out = np.log(a)
    out[np.isnan(out)] = 0.0
    return out


def logsumexp(a, axis=0):
    """Compute the log of the sum of exponentials of input elements.

    Notes
    -----
    Unlike the versions implemented in ``scipy.misc`` and
    ``sklearn.utils.extmath`` this version explicitly masks the underflows
    occured during ``np.exp``.

    Examples
    --------
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    out = np.log(exp_mask_zero(a - a_max).sum(axis=0))
    out += a_max
    return out


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]
