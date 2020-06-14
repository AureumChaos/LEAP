"""
    Unit tests for mutation-related functionality.
"""
from leap_ec import core
from leap_ec import binary_problems
from leap_ec import ops


def test_mutate_bitflip():
    # Create a very simple individual with two binary genes of all ones.
    ind = [core.Individual([1, 1], decoder=core.IdentityDecoder(),
                           problem=binary_problems.MaxOnes())]

    # Now mutate the individual such that we *expect both bits to flip*
    mutated_ind = next(ops.mutate_bitflip(iter(ind), expected=2))

    assert mutated_ind.genome == [0, 0]

    # Of course, since we didn't clone the original, well, that actually got
    # zapped, too.

    assert ind[0].genome == [0, 0]
