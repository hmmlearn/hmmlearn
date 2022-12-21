import numpy as np
from scipy import special


def normalize(a, axis=None):
    """
    Normalize the input array so that it sums to 1.

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
    """
    Normalize the input array so that ``sum(exp(a)) == 1``.

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
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state GMMHMM in the degenerate case normalizing a
        # single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = special.logsumexp(a, axis, keepdims=True)
        a -= a_lse


def fill_covars(covars, covariance_type='full', n_components=1, n_features=1):
    if covariance_type == 'full':
        return covars
    elif covariance_type == 'diag':
        return np.array(list(map(np.diag, covars)))
    elif covariance_type == 'tied':
        return np.tile(covars, (n_components, 1, 1))
    elif covariance_type == 'spherical':
        # Regardless of what is passed in, we flatten in
        # and then expand it to the correct shape
        covars = np.ravel(covars)
        eye = np.eye(n_features)[np.newaxis, :, :]
        covars = covars[:, np.newaxis, np.newaxis]
        return eye * covars
