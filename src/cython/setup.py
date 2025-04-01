from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'AFT Survival Tree',
    ext_modules = cythonize(["distribution_cy.pyx", "tree_cy.pyx"]),
    include_dirs=[numpy.get_include()]
)