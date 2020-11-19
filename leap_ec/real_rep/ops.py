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


@curry
def perform_mutation_guassian(genome: list,
                              expected_num_mutations: float = 1) -> Iterator:
    """ Perform actual Gaussian mutation on real-valued genes

    This used to be inside `mutate_bitflip`, but was moved outside it so that
    `leap_ec.segmented.ops.apply_mutation` could directly use this function,
    thus saving us from doing a copy-n-paste of the same code to the segmented
    sub-package.

    :param genome: of real-valued numbers that will potentially be mutated
    :param expected_num_mutations: on average how many mutations are expected
    :return: mutated genom
    """


##############################
# Function mutate_gaussian
##############################
def mutate_gaussian(std: float, expected_num_mutation: float = None,
                    hard_bounds: Tuple[float, float] = (-math.inf, math.inf)):
    """Mutate and return an individual with a real-valued representation.

    TODO hard_bounds should also be able to take a sequence —Siggy

    :param next_individual: to be mutated

    :param std: standard deviation to be equally applied to all individuals;
        this can be a scalar value or a "shadow vector" of standard deviations

    :param expected_num_mutation: the *expected* number of mutations per individual,
        on average.  If None, all genes will be mutated.

    :param hard_bounds: to clip for mutations; defaults to (- ∞, ∞)
    :return: a generator of mutated individuals.
    """
    def add_gauss(x, std, probability):
        if random.random() < probability:
            return random.gauss(x, std)
        else:
            return x

    def clip(x):
        return max(hard_bounds[0], min(hard_bounds[1], x))

    @iteriter_op
    def mutate(next_individual: Iterator) -> Iterator:
        while True:
            individual = next(next_individual)

            # compute actual probability of mutation based on expected number of
            # mutations and the genome length
            if expected_num_mutation is None:
                # Default to expected probablity of 1.0
                p = compute_expected_probability(1.0, individual.genome)
            else:
                p = compute_expected_probability(expected_num_mutation,
                                                 individual.genome)

            if util.is_sequence(std):
                # We're given a vector of "shadow standard deviations" so apply
                # each sigma individually to each gene
                individual.genome = [
                    clip(
                        add_gauss(
                            x, s, p)) for x, s in zip(
                        individual.genome, std)]
            else:
                individual.genome = [clip(add_gauss(x, std, p))
                                     for x in individual.genome]
            # invalidate fitness since we have new genome
            individual.fitness = None

            yield individual

    return mutate
