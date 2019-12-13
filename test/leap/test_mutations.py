"""
    Unit tests for mutation-related functionality.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from leap import core
from leap import binary_problems
from leap import ops


def test_mutate_bitflip():

    # Create a very simple individual with two binary genes of all ones.
    ind = core.Individual([1, 1], decoder=core.IdentityDecoder(),
                          problem=binary_problems.MaxOnes())

    # Simulate value further up the pipeline for the test
    previous_value = (ind, (), {})

    # Now mutate the individual such that we *expect both bits to flip*
    mutated_ind, args, kwargs = next(ops.mutate_bitflip(iter(previous_value), expected=2))

    assert mutated_ind.genome == [0,0]

    # Of course, since we didn't clone the original, well, that actually got
    # zapped, too.

    assert ind.genome == [0,0]
