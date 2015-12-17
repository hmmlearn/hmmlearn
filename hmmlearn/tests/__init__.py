import numpy as np
from sklearn.datasets.samples_generator import make_spd_matrix

from hmmlearn.utils import normalize

# Make NumPy complain about underflows/overflows etc.
np.seterr(all="warn")


def make_covar_matrix(covariance_type, n_components, n_features):
    mincv = 0.1
    rand = np.random.random
    return {
        'spherical': (mincv + mincv * np.dot(rand((n_components, 1)),
                                             np.ones((1, n_features)))) ** 2,
        'tied': (make_spd_matrix(n_features)
                 + mincv * np.eye(n_features)),
        'diag': (mincv + mincv * rand((n_components, n_features))) ** 2,
        'full': np.array([(make_spd_matrix(n_features)
                           + mincv * np.eye(n_features))
                          for x in range(n_components)])
    }[covariance_type]


def normalized(X, axis=None):
    return normalize(X.copy(), axis=axis)


def log_likelihood_increasing(h, X, lengths, n_iter):
    h.n_iter = 1        # make sure we do a single iteration at a time
    h.init_params = ''  # and don't re-init params
    log_likelihoods = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        h.fit(X, lengths=lengths)
        log_likelihoods[i] = h.score(X, lengths=lengths)

    # TODO: remove == 0, once all the tests are proper.
    diff = np.diff(log_likelihoods) >= 0
    return diff.all()
