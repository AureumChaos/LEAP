**LEAP: Evolutionary Algorithms in Python**

*Written by Dr. Jeffrey K. Bassett, Dr. Mark Coletti, and Eric Scott*

![Build Status](https://travis-ci.org/AureumChaos/LEAP.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/AureumChaos/LEAP/badge.svg?branch=master)](https://coveralls.io/github/AureumChaos/LEAP?branch=master)
[![Documentation Status](https://readthedocs.org/projects/leap-gmu/badge/?version=latest)](https://leap-gmu.readthedocs.io/en/latest/?badge=latest)

LEAP is a general purpose Evolutionary Computation package that combines 
readable and easy-to-use syntax for search and optimization algorithms with 
powerful distribution and visualization features.

LEAP's signature is its operator pipeline, which uses a simple list of 
functional operators to concisely express a metaheuristic algorithm's 
configuration as high-level code.  Adding metrics, visualization, or 
special features (like distribution, coevolution, or island migrations)
is often as simple as adding operators into the pipeline.


# Using LEAP

Get the stable version of LEAP from the Python package index with

```bash
pip install leap_ec
```

## Simple Example

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

## Genetic Algorithm Example

The next-easiest way to use LEAP is to configure a custom algorithm via one 
of the metaheuristic functions in the `leap_ec.algorithms` package.  These 
interfaces offer you a flexible way to customize the various operators, 
representations, and other components that go into a modern evolutionary 
algorithm.

Metaheuristics are usually defined by three main objects: a `Problem`, a
`Representation`, and a pipeline (list) of `Operators`.

Here's an example that applies a genetic algorithm variant to solve the 
`MaxOnes` optimization problem.  It uses bitflip mutation, uniform crossover, 
and binary tournament_selection selection:

```Python
from leap_ec.algorithm import generational_ea
from leap_ec import ops, decoder, representation
from leap_ec.binary_rep import initializers
from leap_ec.binary_rep import problems
from leap_ec.binary_rep.ops import mutate_bitflip

pop_size = 5
ea = generational_ea(generations=10, pop_size=pop_size,

                     # Solve a MaxOnes Boolean optimization problem
                     problem=problems.MaxOnes(),

                     representation=representation.Representation(
                         # Genotype and phenotype are the same for this task
                         decoder=decoder.IdentityDecoder(),
                         # Initial genomes are random binary sequences
                         initialize=initializers.create_binary_sequence(length=10)
                     ),

                     # The operator pipeline
                     pipeline=[ops.tournament_selection,
                               # Select parents via tournament_selection selection
                               ops.clone,  # Copy them (just to be safe)
                               # Basic mutation: defaults to a 1/L mutation rate
                                   mutate_bitflip,
                               # Crossover with a 40% chance of swapping each gene
                               ops.uniform_crossover(p_swap=0.4),
                               ops.evaluate,  # Evaluate fitness
                               # Collect offspring into a new population
                               ops.pool(size=pop_size)
                               ])

print('Generation, Best_Individual')
for i, best in ea:
    print(f"{i}, {best}")
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


