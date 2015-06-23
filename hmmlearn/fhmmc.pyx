cimport cython
cimport numpy as np

import numpy as np
from libc.math cimport exp, log

ctypedef np.float64_t dtype_t
cdef dtype_t _NINF = -np.inf


@cython.boundscheck(False)
cdef inline dtype_t _max(dtype_t[:] values):
    # find maximum value (builtin 'max' is unrolled for speed)
    cdef dtype_t value
    cdef dtype_t vmax = _NINF
    for i in range(values.shape[0]):
        value = values[i]
        if value > vmax:
            vmax = value
    return vmax


@cython.boundscheck(False)
cdef dtype_t _logsumexp(dtype_t[:] X):
    cdef dtype_t vmax = _max(X)
    cdef dtype_t power_sum = 0

    for i in range(X.shape[0]):
        power_sum += exp(X[i] - vmax)

    return log(power_sum) + vmax


@cython.boundscheck(False)
def _forward(int n_observations, int n_chains, int n_states, state_combinations,
        np.ndarray[dtype_t, ndim=2] log_startprob,
        np.ndarray[dtype_t, ndim=3] log_transmat,
        np.ndarray[dtype_t, ndim=2] framelogprob,
        np.ndarray[dtype_t, ndim=3] fwdlattice):
    # TODO: this currently only works for 2 chains since fwdlattice is typed to dim 3!
    #cdef np.ndarray[dtype_t, ndim=n_chains] fwdlattice
    # TODO: make this typed!
    #state_combination_shape = tuple([n_states for _ in xrange(n_chains)])
    #fwdlattice = np.zeros((n_observations,) + state_combination_shape)

    cdef int t, chain_idx, idx, state, k

    # Allocate buffers
    cdef np.ndarray[dtype_t, ndim=1] init_buffer
    init_buffer = np.zeros(n_chains)
    cdef np.ndarray[dtype_t, ndim=1] work_buffer
    work_buffer = np.zeros(n_chains * n_states)

    # Initialize
    for idx, state_combination in enumerate(state_combinations):
        # State probabilities
        for chain_idx, state in enumerate(state_combination):
            init_buffer[chain_idx] = log_startprob[chain_idx][state]

        # Emission probability
        fwdlattice[0][state_combination] = _logsumexp(init_buffer) + framelogprob[0][idx]

    # Forward recursion (naive implementation)
    for t in range(1, n_observations):
        for idx, state_combination in enumerate(state_combinations):
            # State probabilities
            for chain_idx, state in enumerate(state_combination):
                # Here we calculate all previous state combinations that are possible for this specific chain.
                # Since the chains evolve independently, this means that we only have to vary the state
                # of the current chain, hence the chain_index-th entry in the state_combination tuple.
                for k in range(n_states):
                    previous_state_combination = list(state_combination)
                    previous_state_combination[chain_idx] = k
                    previous_state_combination = tuple(previous_state_combination)

                    # previous probability for the current chain and state k * transition probability from state k
                    # to current state. Since all probabilities are logarithmic, addition is the correct operation.
                    prev_logprob = fwdlattice[t - 1][previous_state_combination]
                    trans_logprob = log_transmat[chain_idx][k][state]
                    work_buffer[chain_idx * n_states + k] = prev_logprob + trans_logprob

            # Emission probability
            fwdlattice[t][state_combination] = _logsumexp(work_buffer) + framelogprob[t][idx]
