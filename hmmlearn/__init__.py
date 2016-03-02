"""
hmmlearn
========

``hmmlearn`` is a set of algorithm for learning and inference of
Hiden Markov Models.
"""

__version__ = "0.2.1"


def setup_module(module):
    """Fixture for the tests to assure global seeding of RNGs."""
    import os
    import numpy as np
    import random

    # It could have been provided in the environment.
    _random_seed = os.environ.get("HMMLEARN_SEED")
    if _random_seed is None:
        _random_seed = np.random.uniform() * (2 ** 31 - 1)
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
