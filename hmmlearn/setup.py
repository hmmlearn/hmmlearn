import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('hmmlearn', parent_package, top_path)

    # add cython extension module for hmm
    config.add_extension(
        '_hmmc',
        sources=['_hmmc.c'],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
    )

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
