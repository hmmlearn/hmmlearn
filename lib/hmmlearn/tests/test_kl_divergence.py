import numpy as np
import pytest
import scipy.stats

from hmmlearn import kl_divergence

class TestKLDivergence:

    def test_dirichlet(self):
        v1 = [1, 2, 3, 4]
        v2 = [4, 3, 2, 1]
        assert kl_divergence.kl_dirichlet(v1, v1) == 0
        assert kl_divergence.kl_dirichlet(v2, v2) == 0
        assert kl_divergence.kl_dirichlet(v1, v2) > 0
        assert kl_divergence.kl_dirichlet(v2, v1) > 0

    def test_normal(self):

        assert kl_divergence.kl_normal_distribution(0, 1, 0, 1) == 0
        assert kl_divergence.kl_normal_distribution(0, 1, 1, 1) > 0

    def test_multivariate_normal(self):
        mean_p = [0]
        var_p = [[1]]
        assert kl_divergence.kl_multivariate_normal_distribution(mean_p, var_p, mean_p, var_p) == 0
        mean_q = [1]
        var_q = [[1]]
        assert kl_divergence.kl_multivariate_normal_distribution(mean_p, var_p, mean_q, var_q) > 0

    def test_gamma(self):
        assert kl_divergence.kl_gamma_distribution(1, .01, 1, .01) == 0
        assert kl_divergence.kl_gamma_distribution(1, .01, 2, .01) > 0
        assert kl_divergence.kl_gamma_distribution(1, .01, 1, .02) > 0

    def test_wishart(self):
        dof1 = 952
        scale1 = np.asarray([[339.8474024737109]])
        dof2 = 1.0
        scale2 = np.asarray([[0.001]])

        assert kl_divergence.kl_wishart_distribution(dof1, scale1, dof1, scale1) == 0
        assert kl_divergence.kl_wishart_distribution(dof1, scale1, dof2, scale2) > 0
