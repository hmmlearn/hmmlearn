import numpy as np

from scipy.special import gammaln, digamma
from sklearn.utils import check_array


def kl_dirichlet(q, p):
    """
    KL Divergence between two dirichlet distributions

    KL(q || p) = ln [gamma(q)/gamma(p)] - sum [ ln [gamma(q_j)/gamma(p_j)]
                 - (q_j - p_j) (digamma(q_j) - digamma(p_j)]

    """
    q = np.asarray(q)
    p = np.asarray(p)

    qsum = q.sum()
    psum = p.sum()
    delta = q - p
    return gammaln(qsum) - gammaln(psum) - \
            np.sum(gammaln(q) - gammaln(p)) + \
            np.sum(
                np.dot(
                    delta,
                    (digamma(q) - digamma(qsum))
                )
            )

