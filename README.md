**LEAP: Evolutionary Algorithms in Python**

*Written by Dr. Jeffrey K. Bassett, Dr. Mark Coletti, and Eric Scott*

![Build Status](https://travis-ci.org/AureumChaos/LEAP.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/AureumChaos/LEAP/badge.svg?branch=master)](https://coveralls.io/github/AureumChaos/LEAP?branch=master)
[![Documentation Status](https://readthedocs.org/projects/leap-gmu/badge/?version=latest)](https://leap-gmu.readthedocs.io/en/latest/?badge=latest)

LEAP is a general purpose Evolutionary Computation package that combines readable and easy-to-use syntax for search and
optimization algorithms with powerful <!-- distribution and --> visualization features.


<!-- ## Install with Pip -->

<!-- `pip install leap` -->



## Installing from Source

To get started with LEAP, clone the repo:

```
git clone https://github.com/AureumChaos/LEAP.git
```

and optionally set up a Python virtual environment to isolate its dependencies (this is recommended, but you can typically skip it):

```bash
python -m venv ./venv
source venv/bin/activate
```

Now you can use the Makefile to setup the dependencies and install LEAP as a package.

```bash
make setup
```

All done!  You can now `import leap` in your projects.

Or you can run one of the demo applications from the `example/` directory

```bash
python example/island_models.py
```

![Demo of LEAP running a 3-population island model on a real-valued optimization problem.](_static/island_model_animation.gif)
*Demo of LEAP running a 3-population island model on a real-valued optimization problem.*

## Basic Usage

LEAP's signature is its operator pipeline, which uses a simple list of functional operators to concisely express a
metaheuristic algorithm's configuration as high-level code.

The easiest way to build an EA with LEAP is to use one of the built-in high-level metaheuristics (like 
`generational_ea`) and pass in the operators and components that you want.

Here's an example that applies a genetic algorithm variant to solve the MaxOnes optimization problem.  It uses 
bitflip mutation, uniform crossover, and binary tournament selection:

```Python
from leap.algorithm import generational_ea
from leap import core, ops, binary_problems
l = 10  # The length of the genome
pop_size = 5
ea = generational_ea(generations=100, pop_size=pop_size,
                     individual_cls=core.Individual, # Use the standard Individual as the prototype for the population

                    decoder=core.IdentityDecoder(),          # Genotype and phenotype are the same for this task
                    problem=binary_problems.MaxOnes(),       # Solve a MaxOnes Boolean optimization problem
                    initialize=core.create_binary_sequence(length=10),  # Initial genomes are random binary sequences

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


## Documentation

The stable version of LEAP's full documentation is over at [ReadTheDocs](https://leap_gmu.readthedocs.io/).

If you want to build a fresh set of docs for yourself, you can do so after running `make setup`:

```
make doc
```

This will create HTML documentation in the `docs/build/html/` directory.  It might take a while the first time,
since building the docs involves generating some plots and executing some example algorithms.

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

The output will look something like this:

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
1. Minimally Viable Product -- tentative release in February
    - distributed / parallel fitness evaluations
        - distribute local cores vs. distributed cluster nodes
        - synchronous vs. asynchronous evaluations
    - variable-length genomes
    - parsimony pressure
    - multi-objective optimization
    - minimally complete documentation
        - fleshed out ReadTheDocs documentation
        - technical report
    - checkpoint / restart support
    - hall of fame
1. Future features
    - Rule systems
        - Mich Approach
        - Pitt Approach
    - Genetic Programming (GP)
    - Estimation of Distribution Algorithms (EDA)
        - Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        - Population-based Incremental Learning (PBIL)
        - Bayesian Optimization Algorithm (BOA)
