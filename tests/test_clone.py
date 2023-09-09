"""
    Unit tests for cloning
"""
import numpy as np

from leap_ec.individual import Individual
from leap_ec.decoder import IdentityDecoder
from leap_ec.binary_rep.problems import MaxOnes
import leap_ec.ops as ops


def test_clone():
    # We need an encoder and problem to ensure those float across during
    # clones.
    decoder = IdentityDecoder()
    problem = MaxOnes()

    original = Individual([1, 1], decoder=decoder, problem=problem)

    cloned = next(ops.clone(iter([original])))

    assert original == cloned

    # Yes, but did the other state make it across OK?
    print(original.__dict__)
    print(cloned.__dict__)

    assert original.fitness == cloned.fitness
    assert original.decoder == cloned.decoder
    assert original.problem == cloned.problem
    # use this when comparing complex objects with arrays

    assert original.uuid != cloned.uuid
    assert original.uuid in cloned.parents
