#!/usr/bin/env python3
"""
    Binary representation specific pipeline operators.
"""
from typing import Iterator
import random
from toolz import curry

from .. ops import compute_expected_probability, iteriter_op


##############################
# Function mutate_bitflip
##############################
@curry
@iteriter_op
def mutate_bitflip(next_individual: Iterator, expected: float = 1) -> Iterator:
    """ mutate and return an individual with a binary representation

    >>> from leap_ec import core, binary_problems
    >>> original = Individual([1,1])
    >>> mutated = next(mutate_bitflip(iter([original])))

    :param next_individual: to be mutated
    :param expected: the *expected* number of mutations, on average
    :return: mutated individual
    """
    def flip(gene):
        if random.random() < probability:
            return (gene + 1) % 2
        else:
            return gene

    while True:
        individual = next(next_individual)

        # Given the average expected number of mutations, calculate the
        # probability for flipping each bit.  This calculation must be made
        # each time given that we may be dealing with dynamic lengths.
        probability = compute_expected_probability(expected, individual.genome)

        individual.genome = [flip(gene) for gene in individual.genome]

        individual.fitness = None  # invalidate fitness since we have new genome

        yield individual


