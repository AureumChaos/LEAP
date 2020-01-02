"""
    Unit tests for crossover operators
"""
import sys, os, itertools

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from leap import core
from leap import ops

def test_uniform_crossover():
    pop = []
    pop.append(core.Individual([0, 0]))
    pop.append(core.Individual([1, 1]))

    # We need a cyclic generator because there are only two individuals in the population, and once the first two
    # are selected for uniform crossover, the next two parents are selected and crossed over.  The cyclic iterator
    # ensures we just select the same two individuals again even though the yield statements in the uniform
    # crossover operator are not invoked again.
    i = ops.naive_cyclic_selection_generator(pop)

    # Do swap with 100% certainty, which will cause the two individuals' genomes to exchange values
    new_pop = list(itertools.islice(ops.uniform_crossover(i, p_swap=1.0), 2))
    assert new_pop[0].genome == [1,1]
    assert new_pop[1].genome == [0,0]

    # Note because we didn't clone the selected individuals, *the original population was changed*.
    assert pop[0].genome == [1,1]
    assert pop[1].genome == [0,0]


def test_uniform_crossover_bad_len():
    """ Test assertion for mis-matched genome lengths
    """
    pop = []
    pop.append(core.Individual([0, 0, 1]))
    pop.append(core.Individual([1, 1]))

    i = ops.naive_cyclic_selection_generator(pop)

    with pytest.raises(RuntimeError):
        new_pop = list(itertools.islice(ops.uniform_crossover(i), 2))
