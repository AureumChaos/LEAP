#!/usr/bin/env python3
"""
    Tests for mutliobjective/ops.py
"""
import itertools
from toolz import pipe
import numpy as np
import pandas as pd
import logging
from leap_ec import Individual
from leap_ec.multiobjective.problems import SCHProblem
from leap_ec.multiobjective.ops import fast_nondominated_sort, \
    crowding_distance_calc, rank_ordinal_sort


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
    # We use the Schaffer's problem for Deb et al since that's the simplest
    # benchmark.  It only requires a single gene.
    problem = SCHProblem()
    pop = [Individual(genome=np.array(g), problem=problem) for g in range(-2,3)]
    pop.append(pop[-1].clone())

    pop = Individual.evaluate_population(pop)

    ranks = [
        3, # [4, 16] Note: Shared single-objective fitness with a higher rank
        2, # [1, 9]
        1, # [0, 4]
        1, # [1, 1]
        1, # [4, 0]
        1  # [4, 0] Note: Duplicated fitness
    ]
    
    # 1-1 argsort: 2, 3, 4, 5  [0, 4]
    # 1-2 argsort: 4, 5, 3, 2  [0, 4]
    
    # 2-1 argsort: 1  [1, 1]
    # 2-2 argsort: 1  [9, 9]
    
    # 3-1 argsort: 0  [4, 4]
    # 3-2 argsort: 0  [16, 16]
    
    distances = [
        0, # Edge
        0, # Edge
        np.inf, # Edge
        (4 - 0) / 4 + (4 - 0) / 4,
        np.inf, # Edge
        np.inf  # Edge
    ]

    return pop, ranks, distances


def test_fast_nondominated_sort():
    """ Test for non-dominated sorting """
    pop, ranks, _ = generate_test_pop()
    fast_nondominated_sort(pop)

    np.testing.assert_array_equal(
        [ind.rank for ind in pop],
        ranks
    )

def test_rank_ordinal_sort():
    """ Test for rank ordinal sorting """
    pop, ranks, _ = generate_test_pop()
    rank_ordinal_sort(pop)

    np.testing.assert_array_equal(
        [ind.rank for ind in pop],
        ranks
    )

def test_crowding_distance_calc():
    """ Test of crowding distance calculation """
    pop, _, distances = generate_test_pop()
    fast_nondominated_sort(pop)
    crowding_distance_calc(pop)
    
    np.testing.assert_array_equal(
        [ind.distance for ind in pop],
        distances
    )


def test_sorting_criteria():
    """ Test sorting by rank and distance criteria """
    pop, _, _ = generate_test_pop()
    processed_pop = pipe(pop,
               fast_nondominated_sort,
               crowding_distance_calc)

    sorted_pop = sorted(processed_pop, key=lambda x: (x.rank, -x.distance))

    pass
