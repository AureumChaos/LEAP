#!/usr/bin/env python3
"""
    Pipeline operators for real-valued representations
"""
import math
import random
from typing import Tuple, Iterator
from typing import Iterator, List, Tuple

from toolz import curry

from leap_ec import util
from leap_ec.ops import compute_expected_probability, iteriter_op




##############################
# Function mutate_gaussian
##############################
@curry
@iteriter_op
def mutate_gaussian(next_individual: Iterator,
                    std: float,
                    expected_num_mutations: float = None,
                    hard_bounds: Tuple[float, float] =
                       (-math.inf, math.inf)) -> Iterator:
    """Mutate and return an individual with a real-valued representation.


    >>> from leap_ec.individual import Individual
    >>> from leap_ec.real_rep.ops import mutate_gaussian

    >>> original = Individual([1.0,0.0])
    >>> mutated = next(mutate_gaussian(iter([original]), 1.0))


    TODO hard_bounds should also be able to take a sequence —Siggy

    :param next_individual: to be mutated
    :param std: standard deviation to be equally applied to all individuals;
        this can be a scalar value or a "shadow vector" of standard deviations
    :param expected_num_mutations: the *expected* number of mutations per
        individual, on average.  If None, all genes will be mutated.
    :param hard_bounds: to clip for mutations; defaults to (- ∞, ∞)
    :return: a generator of mutated individuals.
    """
    while True:
        individual = next(next_individual)

        individual.genome = genome_mutate_guassian(individual.genome,
                                                   std,
                                                   expected_num_mutations,
                                                   hard_bounds)

        # invalidate fitness since we have new genome
        individual.fitness = None

        yield individual

    return mutate


@curry
def genome_mutate_guassian(genome: list,
                           std: float,
                           expected_num_mutations: float = 1,
                           hard_bounds: Tuple[float, float] =
                             (-math.inf, math.inf)) -> list:
    """ Perform actual Gaussian mutation on real-valued genes

    This used to be inside `mutate_gaussian`, but was moved outside it so that
    `leap_ec.segmented.ops.apply_mutation` could directly use this function,
    thus saving us from doing a copy-n-paste of the same code to the segmented
    sub-package.

    :param genome: of real-valued numbers that will potentially be mutated
    :param expected_num_mutations: on average how many mutations are expected
    :return: mutated genome
    """
    def add_gauss(x, std, probability):
        if random.random() < probability:
            return random.gauss(x, std)
        else:
            return x

    def clip(x):
        return max(hard_bounds[0], min(hard_bounds[1], x))

    # compute actual probability of mutation based on expected number of
    # mutations and the genome length
    if expected_num_mutations is None:
        # Default to expected probablity of 1.0
        p = compute_expected_probability(1.0, genome)
    else:
        p = compute_expected_probability(expected_num_mutation, genome)

    if util.is_sequence(std):
        # We're given a vector of "shadow standard deviations" so apply
        # each sigma individually to each gene
        genome = [clip(add_gauss(x, s, p))
                             for x, s in zip(genome, std)]
    else:
        genome = [clip(add_gauss(x, std, p))
                             for x in genome]
    return genome