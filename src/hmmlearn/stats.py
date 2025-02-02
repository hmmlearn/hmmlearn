import numpy as np
from scipy import linalg, special
from ._utils import logdet
from .utils import fill_covars


def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """
    Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a
        single data point.

    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.

    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    covariance_type : {"spherical", "diag", "full", "tied"}, optional
        The type of the covariance parameters.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars
    )


def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model."""
    # X: (ns, nf); means: (nc, nf); covars: (nc, nf) -> (ns, nc)
    nc, nf = means.shape
    # Avoid 0 log 0 = nan in degenerate covariance case.
    covars = np.maximum(covars, np.finfo(float).tiny)
    with np.errstate(over="ignore"):
        return -0.5 * (nf * np.log(2 * np.pi)
                       + np.log(covars).sum(axis=-1)
                       + ((X[:, None, :] - means) ** 2 / covars).sum(axis=-1))


def _log_multivariate_normal_density_spherical(X, means, covars):
    """Compute Gaussian log-density at X for a spherical model."""
    nc, nf = means.shape
    if covars.ndim == 1:
        covars = covars[:, np.newaxis]
    covars = np.broadcast_to(covars, (nc, nf))
    return _log_multivariate_normal_density_diag(X, means, covars)


def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model."""
    nc, nf = means.shape
    cv = np.broadcast_to(covars, (nc, nf, nf))
    return _log_multivariate_normal_density_full(X, means, cv)


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    nc, nf = means.shape
    log_prob = []
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(nf),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob.append(-.5 * (nf * np.log(2 * np.pi)
                               + (cv_sol ** 2).sum(axis=1)
                               + cv_log_det))

    return np.transpose(log_prob)


def _variational_log_multivariate_normal_density(X, means, beta, scale, dof,
                                                 covariance_type):
    """
    Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a
        single data point.

    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.

    beta: array_like, shape (n_components, )
        List of n_components estimate on the scale of the variance over
        the means.

    scale : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    dof: array_like, shape (n_components, )
        List of n_components estimate on the scale of the variance over
        the means.

    covariance_type : {"spherical", "diag", "full", "tied"}, optional
        The type of the covariance parameters.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    # Refer to the Gruhl/Sick paper for the notation
    # In general, things are neater if we pretend the covariance is
    # full / tied.  Or, we could treat each case separately, and reduce
    # the number of operations. That's left for the future :-)
    nc, nf = means.shape
    term1 = special.digamma(
        .5 * (dof - np.arange(0, nf)[:, None])
    ).sum(axis=0)
    if covariance_type in ("diag", "spherical"):
        scale = fill_covars(scale, covariance_type, nc, nf)

    W_k = np.linalg.inv(scale)

    term1 += nf * np.log(2) + logdet(W_k)
    term1 /= 2

    # We ignore the constant that is typically excluded in the literature
    term2 = 0
    term3 = nf / beta

    # (X - Means) * W_k * (X-Means)^T * dof
    # shape becomes (nc, n_samples,
    delta = (X - means[:, None])
    # m is the dimension of the mixture
    # i is the length of the sequence X
    # j, k are the n_features
    if covariance_type in ("full", "diag", "spherical"):
       dots = np.einsum("mij,mjk,mik,m->im",
                             delta, W_k, delta, dof)
    else:
       dots = np.einsum("mij,jk,mik,->im",
                             delta, W_k, delta, dof)
    last_term = .5 * (dots + term3)

    return term1 - term2 - last_term
