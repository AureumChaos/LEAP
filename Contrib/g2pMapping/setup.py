from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'g2pEncodingCy',
    ext_modules = cythonize("g2pEncodingCy.pyx"),
)
