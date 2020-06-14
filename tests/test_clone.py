"""
    Unit tests for cloning
"""
from leap_ec import core
from leap_ec import binary_problems
from leap_ec import ops


def test_clone():
    # We need an encoder and problem to ensure those float across during
    # clones.
    decoder = core.IdentityDecoder()
    problem = binary_problems.MaxOnes()

    original = core.Individual([1, 1], decoder=decoder, problem=problem)

    cloned = next(ops.clone(iter([original])))

    assert original == cloned

    # Yes, but did the other state make it across OK?

    assert original.fitness == cloned.fitness
    assert original.decoder == cloned.decoder
    assert original.problem == cloned.problem
    assert original.__dict__ == cloned.__dict__




