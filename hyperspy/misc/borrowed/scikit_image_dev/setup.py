# To create the exectuable
# cmd : python setup.py build_ext --inplace

import os

# import cython
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('transform', parent_package, top_path)
    # config.add_data_dir('tests')
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True)

    cython(['_warps_cy.pyx'], working_path=base_path)
    cython(['_radon_transform.pyx'], working_path=base_path)
    config.add_extension('_warps_cy', sources=['_warps_cy.c'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'])
    config.add_extension('_radon_transform',
                         sources=['_radon_transform.c'],
                         include_dirs=[get_numpy_include_dirs()])
    return config

from numpy.distutils.core import setup
setup(maintainer='scikit-image Developers',
      author='scikit-image Developers',
      maintainer_email='scikit-image@googlegroups.com',
      description='Transforms',
      url='https://github.com/scikit-image/scikit-image',
      license='SciPy License (BSD Style)',
      **(configuration(top_path='').todict()))
