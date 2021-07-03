"""Private utilities."""

import warnings

import numpy as np
from sklearn.utils.validation import NotFittedError


def split_X_lengths(X, lengths):
    if lengths is None:
        return [X]
    else:
        cs = np.cumsum(lengths)
        n_samples = len(X)
        if cs[-1] > n_samples:
            raise ValueError("more than {} samples in lengths array {}"
                             .format(n_samples, lengths))
        elif cs[-1] != n_samples:
            warnings.warn(
                "less that {} samples in lengths array {}; support for "
                "silently dropping samples is deprecated and will be removed"
                .format(n_samples, lengths),
                DeprecationWarning, stacklevel=3)
        return np.split(X, cs)[:-1]


# Copied from scikit-learn 0.19.
def _validate_covars(covars, covariance_type, n_components):
    """Do basic checks on matrix covariance sizes and values."""
    from scipy import linalg
    if covariance_type == 'spherical':
        if len(covars) != n_components:
            raise ValueError("'spherical' covars have length n_components")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be positive")
    elif covariance_type == 'tied':
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif (not np.allclose(covars, covars.T)
              or np.any(linalg.eigvalsh(covars) <= 0)):
            raise ValueError("'tied' covars must be symmetric, "
                             "positive-definite")
    elif covariance_type == 'diag':
        if len(covars.shape) != 2:
            raise ValueError("'diag' covars must have shape "
                             "(n_components, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be positive")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                    or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")


# Copied from scikit-learn 0.19.
def distribute_covar_matrix_to_match_covariance_type(
        tied_cv, covariance_type, n_components):
    """Create all the covariance matrices from a given template."""
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]),
                     (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
    return cv


# Adapted from scikit-learn 0.21.
def check_is_fitted(estimator, attribute):
    if not hasattr(estimator, attribute):
        raise NotFittedError(
            "This %s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this method."
            % type(estimator).__name__)
