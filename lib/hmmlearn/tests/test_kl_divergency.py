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
