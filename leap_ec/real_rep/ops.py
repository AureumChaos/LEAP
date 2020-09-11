#!/usr/bin/env python3
"""
    Pipeline operators for real-valued representations
"""
import math
import random
from typing import Tuple, Iterator

from leap_ec import util

##############################
# Function mutate_gaussian
##############################
def mutate_gaussian(std: float, expected: float = None,
                    hard_bounds: Tuple[float, float] = (-math.inf, math.inf)):
    """ mutate and return an individual with a real-valued representation

    TODO hard_bounds should also be able to take a sequence —Siggy

    :param next_individual: to be mutated

    :param std: standard deviation to be equally applied to all individuals;
        this can be a scalar value or a "shadow vector" of standard deviations

    :param expected: the *expected* number of mutations per individual,
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

    def mutate(next_individual: Iterator) -> Iterator:
        while True:
            individual = next(next_individual)

            # compute actual probability of mutation based on expected number of
            # mutations and the genome length
            if expected is None:
                p = 1.0
            else:
                p = compute_expected_probability(expected, individual.genome)

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
