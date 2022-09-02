#!/usr/bin/env python3
"""
    LEAP pipeline operators for multiobjective optimization.

    For now this just implements NSGA-II, but other multiobjective approaches
    will eventually be included.

    TODO Should we prepend 'nsgaii' in anticipation of adding other functions
    for other MO methods?  Or have an 'nsgaii' sub-sub-package? We have to be
    mindful of backward compatability.
"""
from typing import Iterator
import random
from itertools import chain.from_iterable
from toolz import curry
from math import inf

import numpy as np

from leap_ec.ops import compute_expected_probability, listlist_op, iteriter_op


def tournament_selection():
    """ Tournament selection that takes into consideration rank and crowding
        distance.

        This should be used instead of leap_ec.ops.tournament_selection for
        NSGA-II.

    TODO eventually this will be replaced by using a special key function for
    leap_ec.ops.tournament_selection.
    """
    raise NotImplementedError


##############################
# fast_nondominated_sort operator
##############################
@curry
@listlist_op
def fast_nondominated_sort(population: list, parents: list = None) -> list:
    """ This implements the NSGA-II fast-non-dominated-sort()

    :param population: population to be ranked
    :param parents: optional parents population to be included with the ranking
        process
    """
    if not parents:
        parents = []

    ranks = {1: []}  # rank 1 initially empty

    # First, find rank 1
    for individual in chain.from_iterable(population, parents):
        individual.dominates = []
        individual.dominated_by = 0
        # there is no rank 0, we're just including in initialization as a
        # reality check; i.e., if we see rank zeroes floating around while
        # debugging, we know something went wrong
        individual.rank = 0

        for other_individual in chain.from_iterable(population, parents):
            if individual is other_individual:
                continue
            if individual > other_individual:
                individual.dominates.append(other_individual)
            elif other_individual > individual:
                individual.dominated_by += 1

        if individual.dominated_by == 0:
            individual.rank = 1
            ranks[1].append(individual)

    # Now fill out the remaining ranks
    i = 1
    while ranks.get(i, []) != []:
        next_front = []
        for individual in ranks[i]:
            for other_individual in individual.dominates:
                other_individual.dominated_by -= 1
                if other_individual.dominated_by == 0:
                    other_individual.rank = i + 1
                    next_front.append(other_individual)

        i += 1
        ranks[i] = next_front

    # the parents will have been updated, too, but the next pipeline operator
    # will also look at them
    return population


##############################
# crowding_distance_calc operator
##############################
@curry
@listlist_op
def crowding_distance_calc(population: list, parents: list = None) -> list:
    """ This implements the NSGA-II crowding-distance-assignment()

    :param population: population to calculate crowding distances
    :param parents: optional parents population to be included
    """
    # Bring over a copy of everyone, parents possibly included, because we're
    # going to be sorting them for each objective.
    if not parents:
        entire_pop = population
    else:
        entire_pop = list(chain.from_iterable(population, parents))

    [i.distance = 0 for i in entire_pop]  # init distances to zero to start

    # Presuming this is a population with homogeneous objectives, then we can
    # arbitrarily peep at the first individual's fitness values to determine
    # how many objectives we have.
    num_objectives = population[0].fitness.shape[0]

    for objective in range(num_objectives):
        # sort by objective being mindful that maximization vs. minimization may
        # be different for each objective

        # set first and last elements to infinity

        # update the distance per individuals with a sliding window of
        # three fitnesses for the current objective starting from the second to
        # the second to last individual's

    return population
