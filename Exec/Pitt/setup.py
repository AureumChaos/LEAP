#! /usr/bin/env python

# To use this script, type "python setup.py install"

from distutils.core import setup, Extension
#import os

#cwd = os.getcwd()
#GRNdir = cwd + '/lib-grn-v2.0'

cRuleFuncsModule = Extension('cRuleFuncs',
                      define_macros = [('MAJOR_VERSION', '1'),
                                       ('MINOR_VERSION', '0')],
#                      include_dirs=[GRNdir],
#                      library_dirs=[GRNdir],
#                      libraries=['grn'],
                      sources=['cRuleFuncs.c'])

setup(name='cRuleFuncs',
      version='1.0',
      ext_modules=[cRuleFuncsModule])

