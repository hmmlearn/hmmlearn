import numpy as np
from scipy.misc import logsumexp


def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def log_normalize(a, axis=None):
    """Normalizes the input array so that the exponent of the sum is 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_lse = logsumexp(a, axis)
    a -= a_lse[:, np.newaxis]


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

def log_mask_zero(prob):
    """Computes the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During EM iteration of the model fitting, very small intermediate
    start/transition probabilities could be normalized to Zero, causes
    a RuntimeWarning: divide by zero encountered in log. This function
    masks this unharmful warning.
    """
    prob = np.asarray(prob)
    with np.errstate(divide='ignore'):
        _log_prob = np.where(prob > 0, np.log(prob), 0)
    return _log_prob
