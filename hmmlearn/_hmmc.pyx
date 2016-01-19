# cython: boundscheck=False, wraparound=False

cimport cython
from cython cimport view
from numpy.math cimport expl, logl, isinf, INFINITY

import numpy as np

ctypedef double dtype_t


cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
    # Builtin 'max' is unrolled for speed.
    cdef dtype_t X_max = -INFINITY
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]

    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    for i in range(X.shape[0]):
        acc += expl(X[i] - X_max)

    return logl(acc) + X_max


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
    cdef dtype_t logprob
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


def _compute_lneta(int n_samples, int n_components,
        dtype_t[:, :] fwdlattice,
        dtype_t[:, :] log_transmat,
        dtype_t[:, :] bwdlattice,
        dtype_t[:, :] framelogprob,
        dtype_t[:, :, :] lneta):

    cdef dtype_t logprob = _logsumexp(fwdlattice[n_samples - 1])
    cdef int t, i, j

    with nogil:
        for t in range(n_samples - 1):
            for i in range(n_components):
                for j in range(n_components):
                    lneta[t, i, j] = (fwdlattice[t, i]
                                      + log_transmat[i, j]
                                      + framelogprob[t + 1, j]
                                      + bwdlattice[t + 1, j]
                                      - logprob)


def _viterbi(int n_samples, int n_components,
        dtype_t[:] log_startprob,
        dtype_t[:, :] log_transmat,
        dtype_t[:, :] framelogprob):

    cdef int c0, c1, t, max_pos
    cdef dtype_t[:, ::view.contiguous] viterbi_lattice
    cdef int[::view.contiguous] state_sequence
    cdef dtype_t logprob
    cdef dtype_t[:, ::view.contiguous] work_buffer
    cdef dtype_t buf, maxbuf

    # Initialization
    state_sequence_arr = np.empty(n_samples, dtype=np.int32)
    state_sequence = state_sequence_arr
    viterbi_lattice = np.zeros((n_samples, n_components))
    work_buffer = np.empty((n_components, n_components))

    with nogil:
        for c1 in range(n_components):
            viterbi_lattice[0, c1] = log_startprob[c1] + framelogprob[0, c1]

        # Induction
        for t in range(1, n_samples):
            for c0 in range(n_components):
                maxbuf = -INFINITY
                for c1 in range(n_components):
                    buf = log_transmat[c1, c0] + viterbi_lattice[t-1, c1]
                    work_buffer[c0, c1] = buf
                    if buf > maxbuf:
                        maxbuf = buf

                viterbi_lattice[t, c0] = maxbuf + framelogprob[t, c0]

        # Observation traceback
        maxbuf = -INFINITY
        for c1 in range(n_components):
            buf = viterbi_lattice[n_samples - 1, c1]
            if buf > maxbuf:
                maxbuf = buf
                max_pos = c1

        state_sequence[n_samples - 1] = max_pos
        logprob = viterbi_lattice[n_samples - 1, max_pos]

        for t in range(n_samples - 2, -1, -1):
            maxbuf = -INFINITY
            for c1 in range(n_components):
                buf = viterbi_lattice[t, c1] \
                    + log_transmat[c1, state_sequence[t + 1]]
                if buf > maxbuf:
                    maxbuf = buf
                    max_pos = c1
            state_sequence[t] = max_pos

    return state_sequence_arr, logprob
