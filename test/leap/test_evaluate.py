"""
    Unit tests for cloning
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from leap import core
from leap import binary_problems
from leap import ops


def test_simple_evaluate():
    # Let's try evaluating a single individual
    pop = []

    pop.append(core.Individual([1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    evaluated_individual, context = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness == 2
    assert context == {}



def test_evaluate_with_args():
    pop = []
    pop.append(core.Individual([1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    evaluated_individual, context = next(ops.evaluate(iter(pop), foo='bar', baz=42))

    assert evaluated_individual.fitness == 2
    assert context['foo'] == 'bar'
    assert context['baz'] == 42



