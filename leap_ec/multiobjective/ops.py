#!/usr/bin/env python3
"""
    LEAP pipeline operators for multiobjective optimization.

    For now this just implements NSGA-II, but other multiobjective approaches
    will eventually be included.

    TODO Should we prepend 'nsgaii' in anticipation of adding other functions
    for other MO methods?
"""
from typing import Iterator
import random
from toolz import curry

import numpy as np

from leap_ec.ops import compute_expected_probability, listlist_op, iteriter_op

##############################
# fast_nondominated_sort operator
##############################
@curry
@listlist_op
def fast_nondominated_sort(population: list, parents: list = None) -> list:
    """ This does the NSGA-II
    :param population: population to be ranked
    :param parents: optional parents population to be included with the ranking
        process
    """
    raise NotImplementedError


##############################
# crowding_distance_calc operator
##############################
@curry
@listlist_op
def crowding_distance_calc(population: list) -> list:
    raise NotImplementedError


##############################
# sort_by_dominance operator
##############################
@curry
@listlist_op
def sort_by_dominance(population: list) -> list:
    raise NotImplementedError
