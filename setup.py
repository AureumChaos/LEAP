from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='leap_ec',
    version='0.3.0',
    packages=find_packages(),
    license='Academic',
    author='Mark Coletti, Eric Scott, Jeff Bassett',
    author_email='mcoletti@gmail.com',
    description='A general purpose Library for Evolutionary Algorithms in Python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AureumChaos/LEAP',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Academic Free License (AFL)',
        'Operating System :: OS Independent'
    ],
    pythong_requires='>=3.6',
    install_requires=[
        'dask',         # Used for parallel and distributed algorithms
        'distributed',  # Used for parallel and distributed algorithms
        'matplotlib',   # Used in visualizations
        'networkx',     # Used to specify island model topologies
        'numpy',        # Used for vector math
        'pandas',       # Used to process CSV output for probes
        'toolz'         # Used for functional pipelines of operators
    ]
)
