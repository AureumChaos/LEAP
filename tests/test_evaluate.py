"""
    Unit tests for cloning
"""
from leap_ec import core
from leap_ec import binary_problems
from leap_ec import ops


def test_simple_evaluate():
    # Let's try evaluating a single individual
    pop = [core.Individual([1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]

    evaluated_individual = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness == 2


def test_multiple_evaluations():
    # Let's try evaluating a single individual
    pop = [core.Individual([0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
           core.Individual([0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
           core.Individual([1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
           core.Individual([1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]

    evaluated_individuals = core.Individual.evaluate_population(pop)

    # Since this is the MAX ONES problem, we just count the ones ahead of time, and ensure that the actual
    # fitnesses match up.
    expected_fitnesses = [0, 1, 1, 2]

    for individual, fitness in zip(evaluated_individuals, expected_fitnesses):
        assert individual.fitness == fitness

