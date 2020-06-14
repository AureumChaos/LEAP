"""
    Unit tests for cloning
"""
from math import nan
from leap_ec import core
from leap_ec import binary_problems
from leap_ec import ops
from leap_ec import problem


def test_simple_evaluate():
    # Let's try evaluating a single individual
    pop = [core.Individual([1, 1], decoder=core.IdentityDecoder(),
                           problem=binary_problems.MaxOnes())]

    evaluated_individual = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness == 2
    assert evaluated_individual.is_viable is True


class BrokenProblem(problem.ScalarProblem):
    """ Simulates a problem that throws an exception """

    def __init__(self, maximize):
        super().__init__(maximize)

    def evaluate(self, phenome):
        raise RuntimeError('Simulated exception')


def test_broken_evaluate():
    # Test evaluations that throw exception
    pop = [core.Individual([1, 1], decoder=core.IdentityDecoder(),
                           problem=BrokenProblem(True))]

    evaluated_individual = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness is nan
    assert evaluated_individual.is_viable is False
    assert isinstance(evaluated_individual.exception, RuntimeError)


def test_multiple_evaluations():
    # Let's try evaluating a single individual
    pop = [core.Individual([0, 0], decoder=core.IdentityDecoder(),
                           problem=binary_problems.MaxOnes()),
           core.Individual([0, 1], decoder=core.IdentityDecoder(),
                           problem=binary_problems.MaxOnes()),
           core.Individual([1, 0], decoder=core.IdentityDecoder(),
                           problem=binary_problems.MaxOnes()),
           core.Individual([1, 1], decoder=core.IdentityDecoder(),
                           problem=binary_problems.MaxOnes())]

    evaluated_individuals = core.Individual.evaluate_population(pop)

    # Since this is the MAX ONES problem, we just count the ones ahead of
    # time, and ensure that the actual fitnesses match up.
    expected_fitnesses = [0, 1, 1, 2]

    for individual, fitness in zip(evaluated_individuals, expected_fitnesses):
        assert individual.fitness == fitness
