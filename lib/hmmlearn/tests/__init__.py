import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.utils import check_random_state

from hmmlearn.utils import normalize

# Make NumPy complain about underflows/overflows etc.
np.seterr(all="warn")


def make_covar_matrix(covariance_type, n_components, n_features,
                      random_state=None):
    mincv = 0.1
    prng = check_random_state(random_state)
    if covariance_type == 'spherical':
        return (mincv + mincv * prng.random_sample((n_components,))) ** 2
    elif covariance_type == 'tied':
        return (make_spd_matrix(n_features)
                + mincv * np.eye(n_features))
    elif covariance_type == 'diag':
        return (mincv + mincv *
                prng.random_sample((n_components, n_features))) ** 2
    elif covariance_type == 'full':
        return np.array([
            (make_spd_matrix(n_features, random_state=prng)
             + mincv * np.eye(n_features))
            for x in range(n_components)
        ])


def normalized(X, axis=None):
    X_copy = X.copy()
    normalize(X_copy, axis=axis)
    return X_copy


def log_likelihood_increasing(h, X, lengths, n_iter):
    h.n_iter = 1        # make sure we do a single iteration at a time
    h.init_params = ''  # and don't re-init params
    log_likelihoods = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        h.fit(X, lengths=lengths)
        log_likelihoods[i] = h.score(X, lengths=lengths)

    # XXX the rounding is necessary because LL can oscillate in the
    #     fractional part, failing the tests.
    diff = np.round(np.diff(log_likelihoods), 10) >= 0
    return diff.all()
