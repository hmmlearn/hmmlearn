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


def _forward(int n_samples, int n_components,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
             dtype_t[:, :] framelogprob,
             dtype_t[:, :] fwdlattice):

    cdef int t, i, j
    cdef dtype_t[::view.contiguous] work_buffer = np.zeros(n_components)

    with nogil:
        for i in range(n_components):
            fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]

        for t in range(1, n_samples):
            for j in range(n_components):
                for i in range(n_components):
                    work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]

                fwdlattice[t, j] = _logsumexp(work_buffer) + framelogprob[t, j]


def _backward(int n_samples, int n_components,
              dtype_t[:] log_startprob,
              dtype_t[:, :] log_transmat,
              dtype_t[:, :] framelogprob,
              dtype_t[:, :] bwdlattice):

    cdef int t, i, j
    cdef dtype_t[::view.contiguous] work_buffer = np.zeros(n_components)

    with nogil:
        for i in range(n_components):
            bwdlattice[n_samples - 1, i] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[i, j]
                                      + framelogprob[t + 1, j]
                                      + bwdlattice[t + 1, j])
                bwdlattice[t, i] = _logsumexp(work_buffer)


def _compute_log_xi_sum(int n_samples, int n_components,
                        dtype_t[:, :] fwdlattice,
                        dtype_t[:, :] log_transmat,
                        dtype_t[:, :] bwdlattice,
                        dtype_t[:, :] framelogprob,
                        dtype_t[:, :] log_xi_sum):

    cdef int t, i, j
    cdef dtype_t[:, ::view.contiguous] work_buffer = \
        np.full((n_components, n_components), -INFINITY)
    cdef dtype_t logprob = _logsumexp(fwdlattice[n_samples - 1])

    with nogil:
        for t in range(n_samples - 1):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[i, j] = (fwdlattice[t, i]
                                         + log_transmat[i, j]
                                         + framelogprob[t + 1, j]
                                         + bwdlattice[t + 1, j]
                                         - logprob)

            for i in range(n_components):
                for j in range(n_components):
                    log_xi_sum[i, j] = _logaddexp(log_xi_sum[i, j],
                                                  work_buffer[i, j])


def _viterbi(int n_samples, int n_components,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
             dtype_t[:, :] framelogprob):

    cdef int i, j, t, where_from
    cdef dtype_t logprob

    cdef int[::view.contiguous] state_sequence = \
        np.empty(n_samples, dtype=np.int32)
    cdef dtype_t[:, ::view.contiguous] viterbi_lattice = \
        np.zeros((n_samples, n_components))
    cdef dtype_t[::view.contiguous] work_buffer = np.empty(n_components)

    with nogil:
        for i in range(n_components):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]

        # Induction
        for t in range(1, n_samples):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[j, i]
                                      + viterbi_lattice[t - 1, j])

                viterbi_lattice[t, i] = _max(work_buffer) + framelogprob[t, i]

        # Observation traceback
        state_sequence[n_samples - 1] = where_from = \
            _argmax(viterbi_lattice[n_samples - 1])
        logprob = viterbi_lattice[n_samples - 1, where_from]

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                work_buffer[i] = (viterbi_lattice[t, i]
                                  + log_transmat[i, where_from])

            state_sequence[t] = where_from = _argmax(work_buffer)

    return np.asarray(state_sequence), logprob
