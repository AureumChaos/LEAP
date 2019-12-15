"""
    Unit tests for cloning
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from leap import core
from leap import binary_problems
from leap import ops

def test_clone():
    # We need an encoder and problem to ensure those float across during
    # clones.
    decoder = core.IdentityDecoder()
    problem = binary_problems.MaxOnes()

    original = core.Individual([1, 1], decoder=decoder, problem=problem)

    cloned, context = next(ops.clone(iter([original])))

    assert original == cloned

    # Yes, but did the other state make it across OK?

    assert original.fitness == cloned.fitness
    assert original.decoder == cloned.decoder
    assert original.problem == cloned.problem
    assert original.attributes == cloned.attributes



def test_clone_with_context():
    # We need an encoder and problem to ensure those float across during
    # clones.
    decoder = core.IdentityDecoder()
    problem = binary_problems.MaxOnes()

    original = core.Individual([1, 1], decoder=decoder, problem=problem)

    cloned, context = next(ops.clone(iter([original]), foo='bar', baz=42))

    assert original == cloned
    assert context['foo'] == 'bar'
    assert context['baz'] == 42

    # Yes, but did the other state make it across OK?

    assert original.fitness == cloned.fitness
    assert original.decoder == cloned.decoder
    assert original.problem == cloned.problem
    assert original.attributes == cloned.attributes

