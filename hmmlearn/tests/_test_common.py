import numpy as np


def fit_hmm_and_monitor_log_likelihood(h, X, lengths=None, n_iter=1):
    h.n_iter = 1        # make sure we do a single iteration at a time
    h.init_params = ''  # and don't re-init params
    loglikelihoods = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        h.fit(X, lengths=lengths)
        loglikelihoods[i] = h.score(X, lengths=lengths)
    return loglikelihoods
