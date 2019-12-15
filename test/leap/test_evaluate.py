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

    evaluated_individual = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness == 2



def test_multiple_evaluations():
    # Let's try evaluating a single individual
    pop = []

    pop.append(core.Individual([0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    evaluated_individuals  = [individual for individual in ops.evaluate(iter(pop))]

    # Since this is the MAX ONES problem, we just count the ones ahead of time, and ensure that the actual
    # fitnesses match up.
    expected_fitnesses = [0, 1, 1, 2]

    for individual, fitness in zip(evaluated_individuals, expected_fitnesses):
        assert individual.fitness == fitness

