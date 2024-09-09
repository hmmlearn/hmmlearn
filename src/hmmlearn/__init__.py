"""
hmmlearn
========

``hmmlearn`` is a set of algorithms for learning and inference of
Hidden Markov Models.
"""

import importlib.metadata


try:
    __version__ = importlib.metadata.version("hmmlearn")
except ImportError:
    __version__ = "0+unknown"
