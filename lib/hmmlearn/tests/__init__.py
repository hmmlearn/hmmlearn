import numpy as np

import pytest

from sklearn.datasets import make_spd_matrix
from sklearn.utils import check_random_state

from hmmlearn.utils import normalize
from hmmlearn.base import DECODER_ALGORITHMS

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


def assert_log_likelihood_increasing(h, X, lengths, n_iter):
    h.n_iter = 1        # make sure we do a single iteration at a time
    h.init_params = ''  # and don't re-init params
    log_likelihoods = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        h.fit(X, lengths=lengths)
        log_likelihoods[i] = h.score(X, lengths=lengths)

    # XXX the rounding is necessary because LL can oscillate in the
    #     fractional part, failing the tests.
    diff = np.diff(log_likelihoods)
    value = np.finfo(float).eps ** (1/2)
    assert diff.max() > value, f"Non-increasing log-likelihoods:\n" \
                               f"lls={log_likelihoods}\n" \
                               f"diff={diff}\n" \
                               f"diff.max() < value={diff.min() < value}\n" \
                               f"np.finfo(float).eps={value}\n"


def compare_variational_and_em_models(variational, em, sequences, lengths):
    em_score = em.score(sequences, lengths)
    vi_score = variational.score(sequences, lengths)
    em_scores = em.predict(sequences, lengths)
    vi_scores = variational.predict(sequences, lengths)
    assert em_score == pytest.approx(vi_score), (em_score, vi_score)
    assert np.all(em_scores == vi_scores)

    for decode_algo in DECODER_ALGORITHMS:
        em_logprob, em_path = em.decode(sequences, lengths,
                                        algorithm=decode_algo)
        vi_logprob, vi_path = variational.decode(sequences, lengths,
                                                 algorithm=decode_algo)
        assert em_logprob == pytest.approx(vi_logprob), decode_algo
        assert np.all(em_path == vi_path), decode_algo

    em_predict = em.predict(sequences, lengths)
    vi_predict = variational.predict(sequences, lengths)
    assert np.all(em_predict == vi_predict)
    em_logprob, em_posteriors = em.score_samples(sequences, lengths)
    vi_logprob, vi_posteriors = variational.score_samples(sequences, lengths)
    assert em_logprob == pytest.approx(vi_logprob)
    assert np.all(em_posteriors == pytest.approx(vi_posteriors))

    em_obs, em_states = em.sample(100, random_state=42)
    vi_obs, vi_states = variational.sample(100, random_state=42)
    assert np.all(em_obs == vi_obs)
    assert np.all(em_states == vi_states)


def vi_uniform_startprob_and_transmat(model, lengths):
    nc = model.n_components
    model.startprob_prior_ = np.full(nc, 1/nc)
    model.startprob_posterior_ = np.full(nc, 1/nc) * len(lengths)
    model.transmat_prior_ = np.full((nc, nc), 1/nc)
    model.transmat_posterior_ = np.full((nc, nc), 1/nc)*sum(lengths)
    return model
