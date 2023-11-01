"""

All implementations are based upon the following:
    http://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps
"""

import numpy as np
from scipy.special import gammaln, digamma

from . import _utils


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
    return (gammaln(qsum) - gammaln(psum)
            - np.sum(gammaln(q) - gammaln(p))
            + np.einsum("i,i->", (q - p), (digamma(q) - digamma(qsum))))


def kl_normal_distribution(mean_q, variance_q, mean_p, variance_p):
    """KL Divergence between two normal distributions."""
    result = ((np.log(variance_p / variance_q)) / 2
              + ((mean_q - mean_p)**2 + variance_q) / (2 * variance_p)
              - .5)
    assert result >= 0, result
    return result


def kl_multivariate_normal_distribution(mean_q, covar_q, mean_p, covar_p):
    """
    KL Divergence of two Multivariate Normal Distribtuions

    q(x) = Normal(x; mean_q, variance_q)
    p(x) = Normal(x; mean_p, variance_p)
    """

    # Ensure arrays
    mean_q = np.asarray(mean_q)
    covar_q = np.asarray(covar_q)
    mean_p = np.asarray(mean_p)
    covar_p = np.asarray(covar_p)

    # Need the precision of distribution p
    precision_p = np.linalg.inv(covar_p)

    mean_diff = mean_q - mean_p
    D = mean_q.shape[0]

    # These correspond to the four terms in the ~wpenny paper documented above
    return .5 * (_utils.logdet(covar_p) - _utils.logdet(covar_q)
                 + np.trace(precision_p @ covar_q)
                 + mean_diff @ precision_p @ mean_diff
                 - D)


def kl_gamma_distribution(b_q, c_q, b_p, c_p):
    """
    KL Divergence between two gamma distributions

    q(x) = Gamma(x; b_q, c_q)
    p(x) = Gamma(x; b_p, c_p)
    """
    result = ((b_q - b_p) * digamma(b_q)
              - gammaln(b_q) + gammaln(b_p)
              + b_p * (np.log(c_q) - np.log(c_p))
              + b_q * (c_p-c_q) / c_q)
    assert result >= 0, result
    return result


def kl_wishart_distribution(dof_q, scale_q, dof_p, scale_p):
    """
    KL Divergence between two Wishart Distributions

    q(x) = Wishart(R|dof_q, scale_q)
    p(x) = Wishart(R|dof_p, scale_p)

    Definition from:
        Shihao Ji, B. Krishnapuram, and L. Carin,
        "Variational Bayes for continuous hidden Markov models and its
        application to active learning," IEEE Transactions on Pattern
        Analysis and Machine Intelligence, vol. 28, no. 4, pp. 522–532,
        Apr. 2006, doi: 10.1109/TPAMI.2006.85.
    """
    scale_q = np.asarray(scale_q)
    scale_p = np.asarray(scale_p)
    D = scale_p.shape[0]
    return ((dof_q - dof_p)/2 * _E(dof_q, scale_q)
            - D * dof_q / 2
            + dof_q / 2 * np.trace(scale_p @ np.linalg.inv(scale_q))
            # Division of logarithm turned into subtraction here
            + _logZ(dof_p, scale_p)
            - _logZ(dof_q, scale_q))


def _E(dof, scale):
    r"""
    $L(a, B) = \int \mathcal{Wishart}(\Gamma; a, B) \log |\Gamma| d\Gamma$
    """
    return (-_utils.logdet(scale / 2)
            + digamma((dof - np.arange(scale.shape[0])) / 2).sum())


def _logZ(dof, scale):
    D = scale.shape[0]
    return ((D * (D - 1) / 4) * np.log(np.pi)
            - dof / 2 * _utils.logdet(scale / 2)
            + gammaln((dof - np.arange(scale.shape[0])) / 2).sum())
