import functools
import inspect
import warnings

import numpy as np
from scipy import special
from scipy.stats import multinomial, poisson
from sklearn.utils import check_random_state

from .base import BaseHMM, _AbstractHMM
from .stats import log_multivariate_normal_density
from .utils import fill_covars, log_normalize


_CATEGORICALHMM_DOC_SUFFIX = """

Notes
-----
Unlike other HMM classes, `CategoricalHMM` ``X`` arrays have shape
``(n_samples, 1)`` (instead of ``(n_samples, n_features)``).  Consider using
`sklearn.preprocessing.LabelEncoder` to transform your input to the right
format.
"""


def _make_wrapper(func):
    return functools.wraps(func)(lambda *args, **kwargs: func(*args, **kwargs))


class BaseCategoricalHMM(_AbstractHMM):

    def __init_subclass__(cls):
        for name in [
                "decode",
                "fit",
                "predict",
                "predict_proba",
                "sample",
                "score",
                "score_samples",
        ]:
            meth = getattr(cls, name)
            doc = inspect.getdoc(meth)
            if doc is None or _CATEGORICALHMM_DOC_SUFFIX in doc:
                wrapper = meth
            else:
                wrapper = _make_wrapper(meth)
                wrapper.__doc__ = (
                    doc.replace("(n_samples, n_features)", "(n_samples, 1)")
                    + _CATEGORICALHMM_DOC_SUFFIX)
            setattr(cls, name, wrapper)

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a categorical distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if self.n_features is not None:
            if self.n_features - 1 < X.max():
                raise ValueError(
                    f"Largest symbol is {X.max()} but the model only emits "
                    f"symbols up to {self.n_features - 1}")
        else:
            self.n_features = X.max() + 1

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _compute_likelihood(self, X):
        if X.shape[1] != 1:
            warnings.warn("Inputs of shape other than (n_samples, 1) are "
                          "deprecated.", DeprecationWarning)
            X = np.concatenate(X)[:, None]
        return self.emissionprob_[:, X.squeeze(1)].T

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats=stats, X=X,
                                                  lattice=lattice,
                                                  posteriors=posteriors,
                                                  fwdlattice=fwdlattice,
                                                  bwdlattice=bwdlattice)

        if 'e' in self.params:
            if X.shape[1] != 1:
                warnings.warn("Inputs of shape other than (n_samples, 1) are "
                              "deprecated.", DeprecationWarning)
                X = np.concatenate(X)[:, None]
            np.add.at(stats['obs'].T, X.squeeze(1), posteriors)

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]


class BaseGaussianHMM(_AbstractHMM):

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": {
                "spherical": nc,
                "diag": nc * nf,
                "full": nc * nf * (nf + 1) // 2,
                "tied": nf * (nf + 1) // 2,
            }[self.covariance_type],
        }

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats=stats, X=X,
                                                  lattice=lattice,
                                                  posteriors=posteriors,
                                                  fwdlattice=fwdlattice,
                                                  bwdlattice=bwdlattice)

        if self._needs_sufficient_statistics_for_mean():
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += posteriors.T @ X

        if self._needs_sufficient_statistics_for_covars():
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += posteriors.T @ X**2
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, X, X)

    def _needs_sufficient_statistics_for_mean(self):
        """
        Whether the sufficient statistics needed to update the means are
        updated during calls to `fit`.
        """
        raise NotImplementedError("Must be overriden in subclass")

    def _needs_sufficient_statistics_for_covars(self):
        """
        Whhether the sufficient statistics needed to update the covariances are
        updated during calls to `fit`.
        """
        raise NotImplementedError("Must be overriden in subclass")

    def _generate_sample_from_state(self, state, random_state):
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )


class BaseGMMHMM(BaseHMM):

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nm * nf,
            "c": {
                "spherical": nc * nm,
                "diag": nc * nm * nf,
                "full": nc * nm * nf * (nf + 1) // 2,
                "tied": nc * nf * (nf + 1) // 2,
            }[self.covariance_type],
            "w": nm - 1,
        }

    def _compute_log_weighted_gaussian_densities(self, X, i_comp):
        cur_means = self.means_[i_comp]
        cur_covs = self.covars_[i_comp]
        if self.covariance_type == 'spherical':
            cur_covs = cur_covs[:, None]
        log_cur_weights = np.log(self.weights_[i_comp])

        return log_multivariate_normal_density(
            X, cur_means, cur_covs, self.covariance_type
        ) + log_cur_weights

    def _compute_log_likelihood(self, X):
        logprobs = np.empty((len(X), self.n_components))
        for i in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, i)
            with np.errstate(under="ignore"):
                logprobs[:, i] = special.logsumexp(log_denses, axis=1)
        return logprobs

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post_mix_sum'] = np.zeros((self.n_components, self.n_mix))
        stats['post_sum'] = np.zeros(self.n_components)

        if 'm' in self.params:
            lambdas, mus = self.means_weight, self.means_prior
            stats['m_n'] = lambdas[:, :, None] * mus
        if 'c' in self.params:
            stats['c_n'] = np.zeros_like(self.covars_)

        # These statistics are stored in arrays and updated in-place.
        # We accumulate chunks of data for multiple sequences (aka
        # multiple frames) during fitting. The fit(X, lengths) method
        # in the BaseHMM class will call
        # _accumulate_sufficient_statistics once per sequence in the
        # training samples. Data from all sequences needs to be
        # accumulated and fed into _do_mstep.
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, lattice,
                                          post_comp, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, post_comp, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape

        # Statistics shapes:
        # post_comp_mix     (n_samples, n_components, n_mix)
        # samples           (n_samples, n_features)
        # centered          (n_samples, n_components, n_mix, n_features)

        post_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for p in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, p)
            log_normalize(log_denses, axis=-1)
            with np.errstate(under="ignore"):
                post_mix[:, p, :] = np.exp(log_denses)

        with np.errstate(under="ignore"):
            post_comp_mix = post_comp[:, :, None] * post_mix

        stats['post_mix_sum'] += post_comp_mix.sum(axis=0)
        stats['post_sum'] += post_comp.sum(axis=0)

        if 'm' in self.params:  # means stats
            stats['m_n'] += np.einsum('ijk,il->jkl', post_comp_mix, X)

        if 'c' in self.params:  # covariance stats
            centered = X[:, None, None, :] - self.means_

            def outer_f(x):  # Outer product over features.
                return x[..., :, None] * x[..., None, :]

            if self.covariance_type == 'full':
                centered_dots = outer_f(centered)
                c_n = np.einsum('ijk,ijklm->jklm', post_comp_mix,
                                centered_dots)
            elif self.covariance_type == 'diag':
                centered2 = np.square(centered, out=centered)  # reuse
                c_n = np.einsum('ijk,ijkl->jkl', post_comp_mix, centered2)
            elif self.covariance_type == 'spherical':
                # Faster than (x**2).sum(-1).
                centered_norm2 = np.einsum('...i,...i', centered, centered)
                c_n = np.einsum('ijk,ijk->jk', post_comp_mix, centered_norm2)
            elif self.covariance_type == 'tied':
                centered_dots = outer_f(centered)
                c_n = np.einsum('ijk,ijklm->jlm', post_comp_mix, centered_dots)

            stats['c_n'] += c_n

    def _generate_sample_from_state(self, state, random_state):
        cur_weights = self.weights_[state]
        i_gauss = random_state.choice(self.n_mix, p=cur_weights)
        if self.covariance_type == 'tied':
            # self.covars_.shape == (n_components, n_features, n_features)
            # shouldn't that be (n_mix, ...)?
            covs = self.covars_
        else:
            covs = self.covars_[:, i_gauss]
            covs = fill_covars(covs, self.covariance_type,
                               self.n_components, self.n_features)
        return random_state.multivariate_normal(
            self.means_[state, i_gauss], covs[state]
        )


class BaseMultinomialHMM(BaseHMM):

    def _check_and_set_n_features(self, X):  # Also sets n_trials.
        """
        Check if ``X`` is a sample from a multinomial distribution, i.e. an
        array of non-negative integers, summing up to n_trials.
        """
        super()._check_and_set_n_features(X)
        if not np.issubdtype(X.dtype, np.integer) or X.min() < 0:
            raise ValueError("Symbol counts should be nonnegative integers")
        if self.n_trials is None:
            self.n_trials = X.sum(axis=1)
        elif not (X.sum(axis=1) == self.n_trials).all():
            raise ValueError("Total count for each sample should add up to "
                             "the number of trials")

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _compute_likelihood(self, X):
        probs = np.empty((len(X), self.n_components))
        n_trials = X.sum(axis=1)
        for c in range(self.n_components):
            probs[:, c] = multinomial.pmf(
                X, n=n_trials, p=self.emissionprob_[c, :])
        return probs

    def _compute_log_likelihood(self, X):
        logprobs = np.empty((len(X), self.n_components))
        n_trials = X.sum(axis=1)
        for c in range(self.n_components):
            logprobs[:, c] = multinomial.logpmf(
                X, n=n_trials, p=self.emissionprob_[c, :])
        return logprobs

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            stats['obs'] += posteriors.T @ X

    def _generate_sample_from_state(self, state, random_state):
        try:
            n_trials, = np.unique(self.n_trials)
        except ValueError:
            raise ValueError("For sampling, a single n_trials must be given")
        return multinomial.rvs(n=n_trials, p=self.emissionprob_[state, :],
                               random_state=random_state)


class BasePoissonHMM(BaseHMM):

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "l": nc * nf,
        }

    def _compute_likelihood(self, X):
        probs = np.empty((len(X), self.n_components))
        for c in range(self.n_components):
            probs[:, c] = poisson.pmf(X, self.lambdas_[c]).prod(axis=1)
        return probs

    def _compute_log_likelihood(self, X):
        logprobs = np.empty((len(X), self.n_components))
        for c in range(self.n_components):
            logprobs[:, c] = poisson.logpmf(X, self.lambdas_[c]).sum(axis=1)
        return logprobs

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, lattice,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, obs, lattice, posteriors, fwdlattice, bwdlattice)
        if 'l' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += posteriors.T @ obs

    def _generate_sample_from_state(self, state, random_state):
        return random_state.poisson(self.lambdas_[state])
