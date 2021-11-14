# cython: language_level=3, boundscheck=False, wraparound=False

from cython cimport view
from numpy.math cimport expl, logl, log1pl, isinf, fabsl, INFINITY

import numpy as np

ctypedef double dtype_t


cdef inline int _argmax(dtype_t[:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos


cdef inline dtype_t _max(dtype_t[:] X) nogil:
    return X[_argmax(X)]


cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
    cdef dtype_t X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    for i in range(X.shape[0]):
        acc += expl(X[i] - X_max)

    return logl(acc) + X_max


cdef inline dtype_t _logaddexp(dtype_t a, dtype_t b) nogil:
    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + log1pl(expl(-fabsl(a - b)))


def _forward_log(int n_samples, int n_components,
                 dtype_t[:] log_startprob,
                 dtype_t[:, :] log_transmat,
                 dtype_t[:, :] framelogprob,
                 dtype_t[:, :] fwdlattice):
    """
    Compute the fwdlattice (alpha in the literature)
    probabilities using logarithms:
        P(O_1, O_2, ..., O_t, q_t=S_i | model)
    """

    cdef int t, i, j
    cdef dtype_t[::view.contiguous] tmp_buf = np.zeros(n_components)

    with nogil:
        for i in range(n_components):
            fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]

        for t in range(1, n_samples):
            for j in range(n_components):
                for i in range(n_components):
                    tmp_buf[i] = fwdlattice[t-1, i] + log_transmat[i, j]

                fwdlattice[t, j] = _logsumexp(tmp_buf) + framelogprob[t, j]


def _forward_scaling(int n_samples, int n_components,
                     dtype_t[:] startprob,
                     dtype_t[:, :] transmat,
                     dtype_t[:, :] frameprob,
                     dtype_t[:, :] fwdlattice,
                     dtype_t[:] scaling_factors,
                     dtype_t min_scaling=1e-300):
    """
    Compute the fwdlattice (alpha in the literature)
    probabilities using scaling_factors:
        P(O_1, O_2, ..., O_t, q_t=S_i | model)
    """
    cdef Py_ssize_t t, i, j

    scaling_factors[:] = 0
    fwdlattice[:] = 0

    # Compute intial column of fwdlattice
    for i in range(n_components):
        fwdlattice[0, i] = startprob[i] * frameprob[0, i]

    for i in range(n_components):
        scaling_factors[0] += fwdlattice[0, i]
    if scaling_factors[0] < min_scaling:
        return False  # scaling underflow, stop and hope that we can use logs
    else:
        scaling_factors[0] = 1.0 / scaling_factors[0]

    for i in range(n_components):
        fwdlattice[0, i] *= scaling_factors[0]

    # Compute rest of Alpha
    for t in range(1, n_samples):
        for j in range(n_components):
            for i in range(n_components):
                fwdlattice[t, j] += fwdlattice[t-1, i] * transmat[i, j]
            fwdlattice[t, j] *= frameprob[t, j]

        # Scale this fwdlattice
        for i in range(n_components):
            scaling_factors[t] += fwdlattice[t, i]
        if scaling_factors[t] < min_scaling:
            return False
        else:
            scaling_factors[t] = 1.0 / scaling_factors[t]
        for i in range(n_components):
            fwdlattice[t, i] *= scaling_factors[t]

    return True


def _backward_log(int n_samples, int n_components,
                  dtype_t[:] log_startprob,
                  dtype_t[:, :] log_transmat,
                  dtype_t[:, :] framelogprob,
                  dtype_t[:, :] bwdlattice):
    """
    Compute the backward/beta probabilities using logarithms
        P(O_t+1, O_t+2, ..., O_t, q_t=S_i | model)
    """

    cdef int t, i, j
    cdef dtype_t[::view.contiguous] tmp_buf = np.zeros(n_components)

    with nogil:
        for i in range(n_components):
            bwdlattice[n_samples - 1, i] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                for j in range(n_components):
                    tmp_buf[j] = (log_transmat[i, j]
                                  + framelogprob[t+1, j]
                                  + bwdlattice[t+1, j])
                bwdlattice[t, i] = _logsumexp(tmp_buf)

def _backward_scaling(int n_samples, int n_components,
                      dtype_t[:] startprob,
                      dtype_t[:, :] transmat,
                      dtype_t[:, :] frameprob,
                      dtype_t[:] scaling_factors,
                      dtype_t[:, :] bwdlattice):
    """
    Compute the backward/beta probabilities using scaling_factors:
        P(O_t+1, O_t+2, ..., O_t, q_t=S_i | model)
    """
    cdef Py_ssize_t t, i, j

    bwdlattice[:] = 0
    bwdlattice[n_samples - 1, :] = scaling_factors[n_samples - 1]
    for t in range(n_samples -2, -1, -1):
        for j in range(n_components):
            for i in range(n_components):
                bwdlattice[t, j] += (transmat[j, i]
                                     * frameprob[t+1, i]
                                     * bwdlattice[t+1, i])
            bwdlattice[t, j] = scaling_factors[t] * bwdlattice[t, j]


def _compute_log_xi_sum(int n_samples, int n_components,
                        dtype_t[:, :] fwdlattice,
                        dtype_t[:, :] log_transmat,
                        dtype_t[:, :] bwdlattice,
                        dtype_t[:, :] framelogprob,
                        dtype_t[:, :] log_xi_sum):

    cdef int t, i, j
    cdef dtype_t[:, ::view.contiguous] tmp_buf = \
        np.empty((n_components, n_components))
    cdef dtype_t logprob = _logsumexp(fwdlattice[n_samples - 1])

    with nogil:
        for t in range(n_samples - 1):
            for i in range(n_components):
                for j in range(n_components):
                    tmp_buf[i, j] = (fwdlattice[t, i]
                                     + log_transmat[i, j]
                                     + framelogprob[t + 1, j]
                                     + bwdlattice[t + 1, j]
                                     - logprob)

            for i in range(n_components):
                for j in range(n_components):
                    log_xi_sum[i, j] = _logaddexp(log_xi_sum[i, j],
                                                  tmp_buf[i, j])


def _compute_xi_sum_scaling(int n_samples, int n_components,
                                dtype_t[:, :] fwdlattice,
                                dtype_t[:, :] transmat,
                                dtype_t[:, :] bwdlattice,
                                dtype_t[:, :] frameprob,
                                dtype_t[:, :] xi_sum):
    cdef int t, i, j
    cdef dtype_t[:, ::view.contiguous] tmp_buf = \
        np.empty((n_components, n_components))

    with nogil:
        for t in range(n_samples - 1):
            for i in range(n_components):
                for j in range(n_components):
                    tmp_buf[i, j] = (fwdlattice[t, i]
                                     * transmat[i, j]
                                     * frameprob[t+1, j]
                                     * bwdlattice[t+1, j])

            for i in range(n_components):
                for j in range(n_components):
                    xi_sum[i, j] += tmp_buf[i, j]


def _viterbi(int n_samples, int n_components,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
             dtype_t[:, :] framelogprob):

    cdef int i, j, t, prev
    cdef dtype_t logprob

    cdef int[::view.contiguous] state_sequence = \
        np.empty(n_samples, dtype=np.int32)
    cdef dtype_t[:, ::view.contiguous] viterbi_lattice = \
        np.zeros((n_samples, n_components))
    cdef dtype_t[::view.contiguous] tmp_buf = np.empty(n_components)

    with nogil:
        for i in range(n_components):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]

        # Induction
        for t in range(1, n_samples):
            for i in range(n_components):
                for j in range(n_components):
                    tmp_buf[j] = log_transmat[j, i] + viterbi_lattice[t-1, j]

                viterbi_lattice[t, i] = _max(tmp_buf) + framelogprob[t, i]

        # Observation traceback
        state_sequence[n_samples - 1] = prev = \
            _argmax(viterbi_lattice[n_samples - 1])
        logprob = viterbi_lattice[n_samples - 1, prev]

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                tmp_buf[i] = viterbi_lattice[t, i] + log_transmat[i, prev]

            state_sequence[t] = prev = _argmax(tmp_buf)

    return np.asarray(state_sequence), logprob
