#!/usr/bin/env python3
"""
    Tests for mutliobjective/ops.py
"""
import itertools
import numpy as np
from leap_ec import Individual


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
