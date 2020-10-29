#!/usr/bin/env python3
"""
    Binary representation specific pipeline operators.
"""
from typing import Iterator
import random
from toolz import curry

from leap_ec.ops import compute_expected_probability, iteriter_op

##############################
# Function flip
##############################
def flip(gene, probability):
    """ flip a bit given a probablity

        Note that this is also used in segmented bit flip representation.

        :param gene: a single bit to possibly be flipped
        :param probability: how likely to flip `gene`
    """
    if random.random() < probability:
        return (gene + 1) % 2
    else:
        return gene


##############################
# Function mutate_bitflip
##############################
@curry
@iteriter_op
def mutate_bitflip(next_individual: Iterator, expected_prob: float = 1) -> Iterator:
    """ mutate and return an individual with a binary representation

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.binary_rep.ops import mutate_bitflip

    >>> original = Individual([1,1])
    >>> mutated = next(mutate_bitflip(iter([original])))

    :param next_individual: to be mutated
    :param expected_prob: the *expected* number of mutations, on average
    :return: mutated individual
    """


    while True:
        individual = next(next_individual)

        # Given the average expected number of mutations, calculate the
        # probability for flipping each bit.  This calculation must be made
        # each time given that we may be dealing with dynamic lengths.
        probability = compute_expected_probability(expected_prob,
                                                   individual.genome)

        individual.genome = [flip(gene, probability) for gene in individual.genome]

        individual.fitness = None  # invalidate fitness since we have new genome

        yield individual


