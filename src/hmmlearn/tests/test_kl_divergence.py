import numpy as np

from hmmlearn import _kl_divergence as _kl


class TestKLDivergence:

    def test_dirichlet(self):
        v1 = [1, 2, 3, 4]
        v2 = [4, 3, 2, 1]
        assert _kl.kl_dirichlet(v1, v1) == 0
        assert _kl.kl_dirichlet(v2, v2) == 0
        assert _kl.kl_dirichlet(v1, v2) > 0
        assert _kl.kl_dirichlet(v2, v1) > 0

    def test_normal(self):
        assert _kl.kl_normal_distribution(0, 1, 0, 1) == 0
        assert _kl.kl_normal_distribution(0, 1, 1, 1) > 0

    def test_multivariate_normal(self):
        mean_p = [0]
        var_p = [[1]]
        kl_equal = _kl.kl_multivariate_normal_distribution(
            mean_p, var_p,
            mean_p, var_p)
        assert kl_equal == 0
        # Compare with univariate implementation
        uv = _kl.kl_normal_distribution(0, 1, 0, 1)
        assert kl_equal == uv

        mean_q = [1]
        var_q = [[1]]
        kl_ne = _kl.kl_multivariate_normal_distribution(
            mean_p, var_p,
            mean_q, var_q)

        # Compare with univariate implementation
        uv = _kl.kl_normal_distribution(0, 1, 1, 1)
        assert kl_ne == uv

    def test_gamma(self):
        assert _kl.kl_gamma_distribution(1, .01, 1, .01) == 0
        assert _kl.kl_gamma_distribution(1, .01, 2, .01) > 0
        assert _kl.kl_gamma_distribution(1, .01, 1, .02) > 0

    def test_wishart(self):
        dof1 = 952
        scale1 = np.asarray([[339.8474024737109]])
        dof2 = 1.0
        scale2 = np.asarray([[0.001]])

        kl_equal = _kl.kl_wishart_distribution(dof1, scale1, dof1, scale1)
        assert kl_equal == 0
        kl_ne = _kl.kl_wishart_distribution(dof1, scale1, dof2, scale2)
        assert kl_ne > 0
