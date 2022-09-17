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
from itertools import chain
from toolz import curry
from math import inf

import numpy as np

from leap_ec.ops import compute_expected_probability, listlist_op, iteriter_op
from .problems import MultiObjectiveProblem

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

    This is really *binning* the population by ranks.  In any case, the
    returned population will have an attribute, `rank`, that will denote
    the corresponding rank in which it is a member.

    - Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
      "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
      transactions on evolutionary computation 6, no. 2 (2002): 182-197.

    :param population: population to be ranked
    :param parents: optional parents population to be included with the ranking
        process
    :returns: individuals binned by ranks
    """
    # Ensure that we're dealing with a multi-objective Problem.
    assert isinstance(population[0].problem, MultiObjectiveProblem)

    # Have a separate working copy of the populations
    if parents is not None:
        working_pop = population + parents
    else:
        working_pop = population

    ranks = {1: []}  # rank 1 initially empty

    # First, find rank 1
    for individual in working_pop:
        individual.dominates = []
        individual.dominated_by = 0
        # there is no rank 0, we're just including in initialization as a
        # reality check; i.e., if we see rank zeroes floating around while
        # debugging, we know something went wrong
        individual.rank = 0

        for other_individual in working_pop:
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
    return working_pop


##############################
# crowding_distance_calc operator
##############################
@curry
@listlist_op
def crowding_distance_calc(population: list, parents: list = None) -> list:
    """ This implements the NSGA-II crowding-distance-assignment()

    - Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
      "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
      transactions on evolutionary computation 6, no. 2 (2002): 182-197.

    :param population: population to calculate crowding distances
    :param parents: optional parents population to be included
    """
    # Ensure that we're dealing with a multi-objective Problem.
    assert issubclass(MultiObjectiveProblem, population[0].problem)

    # Bring over a copy of everyone, parents possibly included, because we're
    # going to be sorting them for each objective.
    if not parents:
        entire_pop = population
    else:
        entire_pop = list(chain.from_iterable(population, parents))

    # Presuming this is a population with homogeneous objectives, then we can
    # arbitrarily peep at the first individual's fitness values to determine
    # how many objectives we have.
    num_objectives = entire_pop[0].fitness.shape[0]

    # Check if we're maximizing or minimizing; we arbitrarily check the first
    # individual.
    # TODO We *might* have to check on a case by case basis if we have a weird
    # corner case whereby the population has a mix of different problems.  Or,
    # if there is a mix, that they're homogeneous with regards to maximizing.
    # Note that MultiObjectiveProblem.maximize is a numpy array where a -1 or 1
    # signifies whether we're dealing with maximizing or minimizing.
    is_maximizing = entire_pop[0].problem.maximize

    # minimum and maximum fitnesses by objective, so we initialize to the
    # infinities. At first we assume maximization for all of the objectives,
    # but then we fine-tune for minimization in the next step.
    f_min = np.full(num_objectives, np.inf)
    f_max = np.full(num_objectives, np.NINF)

    for objective in num_objectives:
        if is_maximizing[objective] == -1:
            f_min[objective] = np.NINF
            f_max[objective] = np.inf

    # Find ranges of fitness per objective
    for i in entire_pop:
        i.distance = 0 # init distances to zero to start
        for objective in num_objectives: # update fitness ranges
            f_min[objective] = np.min(f_min[objective], i.fitness[objective])
            f_max[objective] = np.max(f_min[objective], i.fitness[objective])

    objective_ranges = f_max - f_min

    sorted_pop = []

    for objective in range(num_objectives):
        # sort by objective being mindful that maximization vs. minimization may
        # be different for each objective
        if is_maximizing[objective] == -1:
            # If we're maximizing in ascending order, that actually means we
            # want descending order since the larger values are fitter.
            sorted_pop = sorted(population, key=lambda ind: - ind.fitness[objective])
        else:
            sorted_pop = sorted(population, key=lambda ind: ind.fitness[objective])

        # set first and last elements to infinity
        sorted_pop[0].distance = sorted_pop[-1].distance = inf

        # update the distance per individuals with a sliding window of
        # three fitnesses for the current objective starting from the second to
        # the second to last individual's
        for i, individual in enumerate(sorted_pop[1:-1]):
            individual.distance = individual.distance + \
                                  (sorted_pop[i+1].fitness[objective] -
                                   sorted_pop[i-1].fitness[objective]) / objective_ranges[objective]

    return sorted_pop
