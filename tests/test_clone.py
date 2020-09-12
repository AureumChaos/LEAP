"""
    Unit tests for cloning
"""
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

    assert original.fitness == cloned.fitness
    assert original.decoder == cloned.decoder
    assert original.problem == cloned.problem
    assert original.__dict__ == cloned.__dict__
