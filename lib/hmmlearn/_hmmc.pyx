# cython: language_level=3, boundscheck=False, wraparound=False

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


def forward_log(dtype_t[:] log_startprob,
                dtype_t[:, :] log_transmat,
                dtype_t[:, :] framelogprob):
    """
    Compute the forward/alpha lattice using logarithms:
        P(O_1, O_2, ..., O_t, q_t=S_i | model)
    """
    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef dtype_t[:, ::1] fwdlattice = np.zeros((ns, nc))
    cdef ssize_t t, i, j
    cdef dtype_t[::1] tmp_buf = np.empty(nc)
    with nogil:
        for i in range(nc):
            fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]
        for t in range(1, ns):
            for j in range(nc):
                for i in range(nc):
                    tmp_buf[i] = fwdlattice[t-1, i] + log_transmat[i, j]
                fwdlattice[t, j] = _logsumexp(tmp_buf) + framelogprob[t, j]
    return np.asarray(fwdlattice)


def forward_scaling(dtype_t[:] startprob,
                    dtype_t[:, :] transmat,
                    dtype_t[:, :] frameprob,
                    dtype_t min_scaling=1e-300):
    """
    Compute the fwdlattice/alpha lattice using scaling_factors:
        P(O_1, O_2, ..., O_t, q_t=S_i | model)
    """
    cdef ssize_t ns = frameprob.shape[0], nc = frameprob.shape[1]
    cdef dtype_t[:, ::1] fwdlattice = np.zeros((ns, nc))
    cdef dtype_t[::1] scaling_factors = np.zeros(ns)
    cdef ssize_t t, i, j

    with nogil:

        # Compute intial column of fwdlattice
        for i in range(nc):
            fwdlattice[0, i] = startprob[i] * frameprob[0, i]
        for i in range(nc):
            scaling_factors[0] += fwdlattice[0, i]
        if scaling_factors[0] < min_scaling:
            raise ValueError("Forward pass failed with underflow, "
                             "consider using implementation='log' instead")
        else:
            scaling_factors[0] = 1.0 / scaling_factors[0]
        for i in range(nc):
            fwdlattice[0, i] *= scaling_factors[0]

        # Compute rest of Alpha
        for t in range(1, ns):
            for j in range(nc):
                for i in range(nc):
                    fwdlattice[t, j] += fwdlattice[t-1, i] * transmat[i, j]
                fwdlattice[t, j] *= frameprob[t, j]
            for i in range(nc):
                scaling_factors[t] += fwdlattice[t, i]
            if scaling_factors[t] < min_scaling:
                raise ValueError("Forward pass failed with underflow, "
                                 "consider using implementation='log' instead")
            else:
                scaling_factors[t] = 1.0 / scaling_factors[t]
            for i in range(nc):
                fwdlattice[t, i] *= scaling_factors[t]

    return np.asarray(fwdlattice), np.asarray(scaling_factors)


def backward_log(dtype_t[:] log_startprob,
                 dtype_t[:, :] log_transmat,
                 dtype_t[:, :] framelogprob):
    """
    Compute the backward/beta lattice using logarithms:
        P(O_t+1, O_t+2, ..., O_t, q_t=S_i | model)
    """
    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef dtype_t[:, ::1] bwdlattice = np.zeros((ns, nc))
    cdef ssize_t t, i, j
    cdef dtype_t[::1] tmp_buf = np.empty(nc)
    with nogil:
        for i in range(nc):
            bwdlattice[ns-1, i] = 0
        for t in range(ns-2, -1, -1):
            for i in range(nc):
                for j in range(nc):
                    tmp_buf[j] = (log_transmat[i, j]
                                  + framelogprob[t+1, j]
                                  + bwdlattice[t+1, j])
                bwdlattice[t, i] = _logsumexp(tmp_buf)
    return np.asarray(bwdlattice)


def backward_scaling(dtype_t[:] startprob,
                     dtype_t[:, :] transmat,
                     dtype_t[:, :] frameprob,
                     dtype_t[:] scaling_factors):
    """
    Compute the backward/beta lattice using scaling_factors:
        P(O_t+1, O_t+2, ..., O_t, q_t=S_i | model)
    """
    cdef ssize_t ns = frameprob.shape[0], nc = frameprob.shape[1]
    cdef dtype_t[:, ::1] bwdlattice = np.zeros((ns, nc))
    cdef ssize_t t, i, j
    with nogil:
        bwdlattice[:] = 0
        bwdlattice[ns-1, :] = scaling_factors[ns-1]
        for t in range(ns-2, -1, -1):
            for j in range(nc):
                for i in range(nc):
                    bwdlattice[t, j] += (transmat[j, i]
                                         * frameprob[t+1, i]
                                         * bwdlattice[t+1, i])
                bwdlattice[t, j] *= scaling_factors[t]
    return np.asarray(bwdlattice)


def compute_log_xi_sum(dtype_t[:, :] fwdlattice,
                       dtype_t[:, :] log_transmat,
                       dtype_t[:, :] bwdlattice,
                       dtype_t[:, :] framelogprob):
    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef dtype_t[:, ::1] log_xi_sum = np.full((nc, nc), -INFINITY)
    cdef int t, i, j
    cdef dtype_t log_xi, logprob = _logsumexp(fwdlattice[ns-1])
    with nogil:
        for t in range(ns-1):
            for i in range(nc):
                for j in range(nc):
                    log_xi = (fwdlattice[t, i]
                              + log_transmat[i, j]
                              + framelogprob[t+1, j]
                              + bwdlattice[t+1, j]
                              - logprob)
                    log_xi_sum[i, j] = _logaddexp(log_xi_sum[i, j], log_xi)
    return np.asarray(log_xi_sum)


def compute_scaling_xi_sum(dtype_t[:, :] fwdlattice,
                           dtype_t[:, :] transmat,
                           dtype_t[:, :] bwdlattice,
                           dtype_t[:, :] frameprob):
    cdef ssize_t ns = frameprob.shape[0], nc = frameprob.shape[1]
    cdef dtype_t[:, ::1] xi_sum = np.zeros((nc, nc))
    cdef int t, i, j
    with nogil:
        for t in range(ns-1):
            for i in range(nc):
                for j in range(nc):
                    xi_sum[i, j] += (fwdlattice[t, i]
                                     * transmat[i, j]
                                     * frameprob[t+1, j]
                                     * bwdlattice[t+1, j])
    return np.asarray(xi_sum)


def viterbi(dtype_t[:] log_startprob,
            dtype_t[:, :] log_transmat,
            dtype_t[:, :] framelogprob):

    cdef ssize_t ns = framelogprob.shape[0], nc = framelogprob.shape[1]
    cdef int i, j, t, prev
    cdef dtype_t logprob
    cdef int[::1] state_sequence = np.empty(ns, dtype=np.int32)
    cdef dtype_t[:, ::1] viterbi_lattice = np.zeros((ns, nc))
    cdef dtype_t[::1] tmp_buf = np.empty(nc)

    with nogil:
        for i in range(nc):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]

        # Induction
        for t in range(1, ns):
            for i in range(nc):
                for j in range(nc):
                    tmp_buf[j] = log_transmat[j, i] + viterbi_lattice[t-1, j]

                viterbi_lattice[t, i] = _max(tmp_buf) + framelogprob[t, i]

        # Observation traceback
        state_sequence[ns-1] = prev = _argmax(viterbi_lattice[ns-1])
        logprob = viterbi_lattice[ns-1, prev]

        for t in range(ns-2, -1, -1):
            for i in range(nc):
                tmp_buf[i] = viterbi_lattice[t, i] + log_transmat[i, prev]

            state_sequence[t] = prev = _argmax(tmp_buf)

    return np.asarray(state_sequence), logprob
