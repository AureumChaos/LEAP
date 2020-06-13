**LEAP: Evolutionary Algorithms in Python**

*Written by Dr. Jeffrey K. Bassett, Dr. Mark Coletti, and Eric Scott*

![Build Status](https://travis-ci.org/AureumChaos/LEAP.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/AureumChaos/LEAP/badge.svg?branch=master)](https://coveralls.io/github/AureumChaos/LEAP?branch=master)
[![Documentation Status](https://readthedocs.org/projects/leap-gmu/badge/?version=latest)](https://leap-gmu.readthedocs.io/en/latest/?badge=latest)

LEAP is a general purpose Evolutionary Computation package that combines readable and easy-to-use syntax for search and
optimization algorithms with powerful distribution and visualization features.

LEAP's signature is its operator pipeline, which uses a simple list of 
functional operators to concisely express a metaheuristic algorithm's 
configuration as high-level code.


# Examples

## Simple Example

Get the stable version of LEAP with

```bash
pip install leap_ec
```

The easiest way to use an evolutionary algorithm in LEAP is to use the 
`leap_ec.simple` package, which contains simple interfaces for pre-built
algorithms:

```Python
from leap_ec.simple import ea_solve

def f(x):
    """A real-valued function to optimized."""
    return sum(x)**2

ea_solve(f, bounds=[(-5.12, 5.12) for _ in range(5)], maximize=True)
```

## Metaheuristic Algorithm Example

The next-easiest way to use LEAP is to configure a custom algorithm via one 
of the metaheuristic functions in the `leap_ec.algorithms`.  These 
interfaces allow you to customize the various operators, representations, 
and other components that go into a modern evolutionary algorithm.

Here's an example that applies a genetic algorithm variant to solve the 
`MaxOnes` optimization problem.  It uses bitflip mutation, uniform crossover, 
and binary tournament selection:

```Python
from leap_ec.algorithm import generational_ea
from leap_ec import core, ops, binary_problems
pop_size = 5
ea = generational_ea(generations=100, pop_size=pop_size,
                     problem=binary_problems.MaxOnes(),             # Solve a MaxOnes Boolean optimization problem
                     
                     representation=core.Representation(
                        decoder=core.IdentityDecoder(),             # Genotype and phenotype are the same for this task
                        initialize=core.create_binary_sequence(length=10)  # Initial genomes are random binary sequences
                     )

                     # The operator pipeline
                     pipeline=[ops.tournament,                     # Select parents via tournament selection
                               ops.clone,                          # Copy them (just to be safe)
                               ops.mutate_bitflip,                 # Basic mutation: defaults to a 1/L mutation rate
                               ops.uniform_crossover(p_swap=0.4),  # Crossover with a 40% chance of swapping each gene
                               ops.evaluate,                       # Evaluate fitness
                               ops.pool(size=pop_size)             # Collect offspring into a new population
                     ])

print(list(ea))
```

## More Examples

A number of LEAP demo applications are found in the the `example/` directory of the github repository:

```bash
git clone https://github.com/AureumChaos/LEAP.git
python LEAP/example/island_models.py
```

![Demo of LEAP running a 3-population island model on a real-valued optimization problem.](_static/island_model_animation.gif)
*Demo of LEAP running a 3-population island model on a real-valued optimization problem.*


# Documentation

The stable version of LEAP's full documentation is over at [ReadTheDocs](https://leap_gmu.readthedocs.io/).

If you want to build a fresh set of docs for yourself, you can do so after running `make setup`:

```
make doc
```

This will create HTML documentation in the `docs/build/html/` directory.  It might take a while the first time,
since building the docs involves generating some plots and executing some example algorithms.


# Installing from Source

To install a source distribution of LEAP, clone the repo:

```
git clone https://github.com/AureumChaos/LEAP.git
```

And use the Makefile to install the package:

```bash
make setup
```

## Run the Test Suite

LEAP ships with a two-part `pytest` harness, divided into fast and slow tests.  You can run them with 

```bash
make test-fast
```
and 

```bash
make test-slow
```

respectively.

![pytest output example](_static/pytest_output.png)




## Roadmap

The LEAP development roadmap is as follows:

1. ~~pre-Minimally Viable Product~~ -- released 1/14/2020 as ``0.1-pre``
    - ~~basic support for binary representations~~
        - ~~bit flip mutation~~
        - ~~point-wise crossover~~
        - ~~uniform crossover~~
    - ~~basic support for real-valued representations~~
        - ~~mutate gaussian~~
    - ~~selection operators~~
        - ~~truncation selection~~
        - ~~tournament selection~~
        - ~~random selection~~
        - ~~deterministic cyclic selection~~
        - ~~insertion selection~~
    - ~~continuous integration via Travis~~
    - ~~common test functions~~
        - ~~binary~~
            - ~~MAXONES~~
        - ~~real-valued, optionally translated, rotated, and scaled~~
            - ~~Ackley~~
            - ~~Cosine~~
            - ~~Griewank~~
            - ~~Langermann~~
            - ~~Lunacek~~
            - ~~Noisy Quartic~~
            - ~~Rastrigin~~
            - ~~Rosenbock~~
            - ~~Schwefel~~
            - ~~Shekel~~
            - ~~Spheroid~~
            - ~~Step~~
            - ~~Weierstrass~~
    - ~~test harnesses~~
        - `pytest` ~~supported~~
    - ~~simple usage examples~~
        - ~~canonical EAs~~
            - ~~genetic algorithms (GA)~~
            - ~~evolutionary programming (EP)~~
            - ~~evolutionary strategies (ES)~~
        - ~~simple island model~~
        - ~~basic run-time visualizations~~
        - ~~use with Jupyter notebooks~~
    - ~~documentation outline/stubs for ReadTheDocs~~
1. Minimally Viable Product -- tentative release in June 2020
    - ~~distributed / parallel fitness evaluations~~
        - ~~distribute local cores vs. distributed cluster nodes~~
        - ~~synchronous vs. asynchronous evaluations~~
    - ~~variable-length genomes~~
    - parsimony pressure
    - multi-objective optimization
    - minimally complete documentation
        - fleshed out ReadTheDocs documentation
        - technical report
    - checkpoint / restart support
    - hall of fame
    - ~~Gray encoding~~
1. Future features, in no particular order of priority
    - Rule systems
        - Mich Approach
        - Pitt Approach
    - Genetic Programming (GP)
    - Estimation of Distribution Algorithms (EDA)
        - Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        - Population-based Incremental Learning (PBIL)
        - Bayesian Optimization Algorithm (BOA)
