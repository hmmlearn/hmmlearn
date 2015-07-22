import numpy as np


def normalize(A, axis=None, mask=None):
    """Normalize the input array so that it sums to 1.

    Parameters
    ----------
    A: array, shape (n_samples, n_features)
        Non-normalized input data.
    axis: int
        Dimension along which normalization is performed.

    Returns
    -------
    normalized_A: array, shape (n_samples, n_features)
        A with values normalized (summing to 1) along the prescribed axis

    WARNING: Modifies the array inplace.
    """
    A += np.finfo(float).eps
    if mask is None:
        inverted_mask = None
    else:
        inverted_mask = np.invert(mask)
        A[inverted_mask] = 0.0
    Asum = A.sum(axis)

    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape

    A /= Asum
    if inverted_mask is not None:
        A[inverted_mask] = np.finfo(float).eps
    # TODO: should return nothing, since the operation is inplace.
    return A
