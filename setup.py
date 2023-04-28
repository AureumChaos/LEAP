from setuptools import setup, find_packages

# Load the version number from inside the package
exec(open('leap_ec/__version__.py').read())

# Use the README as the long_description
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='leap_ec',
    version=__version__,
    packages=find_packages(exclude=["*tests*"]),
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
    python_requires='>=3.7',
    install_requires=[
        'dask',             # Used for parallel and distributed algorithms
        'distributed',      # Used for parallel and distributed algorithms
        'docopt',           # Elegant command-line interfaces
        'matplotlib',       # Used in visualizations
        'networkx',         # Used to specify island model topologies
        'numpy',            # Used for vector math
        'omegaconf',        # Used for YAML config files
        'pandas',           # Used to process CSV output for probes
        'rich',             # Used for pretty printing logs etc.
        'scipy',
        'toolz'             # Used for functional pipelines of operators
    ]
)
