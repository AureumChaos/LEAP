from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='leap-ec',
    version='0.1.0',
    packages=find_packages(),
    license='Academic',
    author='Jeff Bassett, Eric Scott, Mark Coletti',
    author_email='jbassett.2@gmail.com',
    description='A general purpose Library for Evolutionary Algorithms in Python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AureumChaos/LEAP',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Academic Free License (AFL)',
        'Operating System :: OS Independent'
    ],
    pythong_requires='>=3.6'
)
