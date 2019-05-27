from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'g2pDecoderCy',
    ext_modules = cythonize("g2pDecoderCy.pyx"),
)
