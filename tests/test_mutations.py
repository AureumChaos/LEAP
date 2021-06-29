"""
    Unit tests for mutation-related functionality.
"""
import numpy as np

from leap_ec.individual import Individual
from leap_ec.binary_rep.problems import MaxOnes
import leap_ec.binary_rep.ops as ops


def test_mutate_bitflip():
    # Create a very simple individual with two binary genes of all ones.
    ind = [Individual(np.array([1, 1]),
                      problem=MaxOnes())]

    # Now mutate the individual such that we *expect both bits to bitflip*
    mutated_ind = next(ops.mutate_bitflip(iter(ind), expected_num_mutations=2))

    assert np.all(mutated_ind.genome == [0, 0])

    # Of course, since we didn't clone the original, well, that actually got
    # zapped, too.

    assert np.all(ind[0].genome == [0, 0])
