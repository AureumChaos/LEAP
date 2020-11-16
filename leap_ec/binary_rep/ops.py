#!/usr/bin/env python3
"""
    Binary representation specific pipeline operators.
"""
from typing import Iterator
import random
from toolz import curry

from leap_ec.ops import compute_expected_probability, iteriter_op


##############################
# Function mutate_bitflip
##############################
@curry
@iteriter_op
def mutate_bitflip(next_individual: Iterator,
                   expected_num_mutations: float = 1) -> Iterator:
    """Perform bit-flip mutation on each individual in an iterator (population).

    This assumes that the genomes have a binary representation.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.binary_rep.ops import mutate_bitflip

    >>> original = Individual([1,1])
    >>> mutated = next(mutate_bitflip(iter([original])))

    :param next_individual: to be mutated
    :param expected_num_mutations: the *expected* number of mutations,
        on average
    :return: mutated individual
    """
    while True:
        individual = next(next_individual)

        individual.genome = individual_mutate_bitflip(individual.genome,
                                                   expected_num_mutations=expected_num_mutations)

        individual.fitness = None  # invalidate fitness since we have new genome

        yield individual


##############################
# Function perform_mutate_bitflip
##############################
@curry
def individual_mutate_bitflip(genome: Iterator,
                           expected_num_mutations: float = 1) -> Iterator:
    """Perform bitflip mutation on a particular genome.

    This function can be used by more complex operators to mutate a full population
    (as in `mutate_bitflip`), to work with genome segments (as in
    `leap_ec.segmented.ops.apply_mutation`), etc.  This way we don't have to 
    copy-and-paste the same code for related operators.

    :param genome: of binary digits that we will be mutating
    :param expected_num_mutations: on average how many mutations are we expecting?
    :return: mutated genome
    """
    def bitflip(bit, probability):
        """ bitflip a bit given a probability
        """
        if random.random() < probability:
            return (bit + 1) % 2
        else:
            return bit

    # Given the average expected number of mutations, calculate the
    # probability for flipping each bit.  This calculation must be made
    # each time given that we may be dealing with dynamic lengths.
    probability = compute_expected_probability(expected_num_mutations,
                                               genome)

    genome = [bitflip(gene, probability) for gene in genome]

    return genome
