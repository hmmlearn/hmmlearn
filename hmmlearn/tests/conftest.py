# -*- coding: utf-8 -*-

def pytest_configure(config):
    import os
    import random

    import numpy as np

    _random_seed = int(os.environ.get("HMMLEARN_SEED",
                                      np.random.uniform() * (2**31 - 1)))

    print("set RNG seed to {0}".format(_random_seed))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
