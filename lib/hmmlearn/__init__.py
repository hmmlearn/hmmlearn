"""
hmmlearn
========

``hmmlearn`` is a set of algorithms for learning and inference of
Hidden Markov Models.
"""

try:
    import setuptools_scm
    __version__ = setuptools_scm.get_version(  # xref setup.py
        root="../..", relative_to=__file__,
        version_scheme="post-release", local_scheme="node-and-date")
except (ImportError, LookupError):
    try:
        from ._version import version as __version__
    except ImportError:
        pass
