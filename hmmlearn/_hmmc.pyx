# cython: boundscheck=False

cimport cython
cimport numpy as np
from numpy.math cimport expl, logl, isinf, INFINITY

import numpy as np

ctypedef np.float64_t dtype_t


cdef dtype_t _logsumexp(dtype_t[:] X) nogil:
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
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        np.ndarray[dtype_t, ndim=2] fwdlattice):

    cdef int t, i, j
    cdef np.ndarray[dtype_t, ndim=1] work_buffer
    work_buffer = np.zeros(n_components)

    for i in range(n_components):
        fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]

    for t in range(1, n_samples):
        for j in range(n_components):
            for i in range(n_components):
                work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]

            fwdlattice[t, j] = _logsumexp(work_buffer) + framelogprob[t, j]


def _backward(int n_samples, int n_components,
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        np.ndarray[dtype_t, ndim=2] bwdlattice):

    cdef int t, i, j
    cdef double logprob
    cdef np.ndarray[dtype_t, ndim = 1] work_buffer
    work_buffer = np.zeros(n_components)

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
        np.ndarray[dtype_t, ndim=2] fwdlattice,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] bwdlattice,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        np.ndarray[dtype_t, ndim=3] lneta):

    cdef dtype_t logprob = _logsumexp(fwdlattice[-1])
    cdef int t, i, j

    for t in range(n_samples - 1):
        for i in range(n_components):
            for j in range(n_components):
                lneta[t, i, j] = (fwdlattice[t, i]
                                  + log_transmat[i, j]
                                  + framelogprob[t + 1, j]
                                  + bwdlattice[t + 1, j]
                                  - logprob)


def _viterbi(int n_samples, int n_components,
        np.ndarray[dtype_t, ndim=1] log_startprob,
        np.ndarray[dtype_t, ndim=2] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob):

    cdef int t, max_pos
    cdef np.ndarray[dtype_t, ndim = 2] viterbi_lattice
    cdef np.ndarray[np.int_t, ndim = 1] state_sequence
    cdef dtype_t logprob
    cdef np.ndarray[dtype_t, ndim = 2] work_buffer

    # Initialization
    state_sequence = np.empty(n_samples, dtype=np.int)
    viterbi_lattice = np.zeros((n_samples, n_components))
    viterbi_lattice[0] = log_startprob + framelogprob[0]

    # Induction
    for t in range(1, n_samples):
        work_buffer = viterbi_lattice[t-1] + log_transmat.T
        viterbi_lattice[t] = np.max(work_buffer, axis=1) + framelogprob[t]

    # Observation traceback
    max_pos = np.argmax(viterbi_lattice[n_samples - 1, :])
    state_sequence[n_samples - 1] = max_pos
    logprob = viterbi_lattice[n_samples - 1, max_pos]

    for t in range(n_samples - 2, -1, -1):
        max_pos = np.argmax(viterbi_lattice[t, :] \
                + log_transmat[:, state_sequence[t + 1]])
        state_sequence[t] = max_pos

    return state_sequence, logprob
