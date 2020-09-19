"""
    Unit tests for cloning
"""
from math import nan

from leap_ec.individual import Individual, RobustIndividual
from leap_ec.decoder import IdentityDecoder

import leap_ec.ops as ops
import leap_ec.problem
from leap_ec.binary_rep.problems import MaxOnes


def test_simple_evaluate():
    # Let's try evaluating a single individual
    pop = [Individual([1, 1], decoder=IdentityDecoder(),
                           problem=MaxOnes())]

    evaluated_individual = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness == 2


def test_simple_robust_evaluate():
    # Let's try evaluating a single individual
    pop = [RobustIndividual([1, 1], decoder=IdentityDecoder(),
                           problem=MaxOnes())]

    evaluated_individual = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness == 2
    assert evaluated_individual.is_viable is True


class BrokenProblem(leap_ec.problem.ScalarProblem):
    """ Simulates a problem that throws an exception """

    def __init__(self, maximize):
        super().__init__(maximize)

    def evaluate(self, phenome):
        raise RuntimeError('Simulated exception')


def test_broken_evaluate():
    # Test evaluations that throw exception
    pop = [RobustIndividual([1, 1], decoder=IdentityDecoder(),
                           problem=BrokenProblem(True))]

    evaluated_individual = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness is nan
    assert evaluated_individual.is_viable is False
    assert isinstance(evaluated_individual.exception, RuntimeError)


def test_multiple_evaluations():
    # Let's try evaluating a single individual
    pop = [Individual([0, 0], decoder=IdentityDecoder(),
                           problem=MaxOnes()),
           Individual([0, 1], decoder=IdentityDecoder(),
                           problem=MaxOnes()),
           Individual([1, 0], decoder=IdentityDecoder(),
                           problem=MaxOnes()),
           Individual([1, 1], decoder=IdentityDecoder(),
                           problem=MaxOnes())]

    evaluated_individuals = Individual.evaluate_population(pop)

    # Since this is the MAX ONES problem, we just count the ones ahead of
    # time, and ensure that the actual fitnesses match up.
    expected_fitnesses = [0, 1, 1, 2]

    for individual, fitness in zip(evaluated_individuals, expected_fitnesses):
        assert individual.fitness == fitness
