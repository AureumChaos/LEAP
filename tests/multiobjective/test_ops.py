#!/usr/bin/env python3
"""
    Tests for mutliobjective/ops.py
"""
import itertools
import numpy as np
from leap_ec import Individual
from leap_ec.multiobjective.problems import SCHProblem
from leap_ec.multiobjective.ops import fast_nondominated_sort, \
    crowding_distance_calc


def test_sort_by_2nd_objective():
    """ Simple test to ensure we can sort by 2nd objective fitness """
    pop = []
    for combo in itertools.combinations_with_replacement('21', 2):
        ind = Individual([])
        ind.fitness = np.array([float(combo[0]), float(combo[1])])
        pop.append(ind)
    new_pop = sorted(pop, key=lambda ind: ind.fitness[1])
    target = [np.array([2.0, 1.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    for i, ind in enumerate(new_pop):
        assert all(ind.fitness == target[i])


def generate_test_pop():
    """ Common mechanism for generating a test population

        Uses the SCH test function as that's a simple benchmark by which
        to do manual confirmation of results.
    """
    # First set up individuals with two genes of combinations of
    # [-2,-1,0,1,2]
    genomes = [np.array(a) for a in itertools.product(range(3),repeat=2)]
    # we start from the second element to skip (0,0) so it isn't duplicated
    # to get all the negatives
    genomes += [a * -1 for a in genomes[1:]]

    # We use the Schaffer's problem for Deb et al since that's the simplest
    # benchmark.
    pop = [Individual(genome=g, problem=SCHProblem()) for g in genomes]

    pop = Individual.evaluate_population(pop)

    return pop


def test_fast_nondominated_sort():
    """ Test for non-dominated sorting """
    pop = generate_test_pop()

    sorted_pop = fast_nondominated_sort(pop)

    # TODO add manual checks to ensure that sorted_pop is binned
    # correctly by ranks.

    pass


def test_crowding_distance_calc():
    """ Test of crowding distance calculation """
    pop = generate_test_pop()

    sorted_pop = crowding_distance_calc(pop)

    pass
