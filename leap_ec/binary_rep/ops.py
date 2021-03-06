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
                   expected_num_mutations: float = None,
                   probability: float = None) -> Iterator:
    """Perform bit-flip mutation on each individual in an iterator (population).

    This assumes that the genomes have a binary representation.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.binary_rep.ops import mutate_bitflip

    >>> original = Individual([1,1])
    >>> op = mutate_bitflip(expected_num_mutations=1)
    >>> pop = iter([original])
    >>> mutated = next(op(pop))

    :param next_individual: to be mutated
    :param expected_num_mutations: on average how many mutations done (specificy either this or probability, but not both)
    :param probability: the probability of mutating any given gene (specificy either this or expected_num_mutations, but not both)
    :return: mutated individual
    """
    if (expected_num_mutations is not None) and (probability is not None):
        raise ValueError("Received parameters for 'probability' and 'expected_num_mutations', but can only use one or the other.")
    if (expected_num_mutations is None) and (probability is None):
        raise ValueError("Received no value for 'probability' or 'expected_num_mutations'.  Must have one.")
    if (probability is not None) and ((probability < 0) or (probability > 1)):
        raise ValueError(f"The value of 'probability' is {probability}, but must be >= 0 and <= 1.")

    while True:
        individual = next(next_individual)

        individual.genome = genome_mutate_bitflip(individual.genome,
                                                      expected_num_mutations=expected_num_mutations,
                                                      probability=probability)

        individual.fitness = None  # invalidate fitness since we have new genome

        yield individual


##############################
# Function perform_mutate_bitflip
##############################
@curry
def genome_mutate_bitflip(genome: list,
                          expected_num_mutations: float = None,
                          probability: float = None) -> list:
    """Perform bitflip mutation on a particular genome.

    This function can be used by more complex operators to mutate a full population
    (as in `mutate_bitflip`), to work with genome segments (as in
    `leap_ec.segmented.ops.apply_mutation`), etc.  This way we don't have to
    copy-and-paste the same code for related operators.

    :param genome: of binary digits that we will be mutating
    :param expected_num_mutations: on average how many mutations are we expecting?
    :return: mutated genome
    """
    assert(bool(expected_num_mutations is not None) ^ bool(probability is not None)), f"Got expected_num_mutations={expected_num_mutations} and probability={probability}.  One must be specified, but not both."
    assert((probability is None) or (probability >= 0))
    assert((probability is None) or (probability <= 1))

    def bitflip(bit, probability):
        """ bitflip a bit given a probability
        """
        if random.random() < probability:
            return (bit + 1) % 2
        else:
            return bit

    if probability is None:
        # Given the average expected number of mutations, calculate the
        # probability for flipping each bit.  This calculation must be made
        # each time given that we may be dealing with dynamic lengths.
        p = compute_expected_probability(expected_num_mutations, genome)
    else:
        p = probability

    genome = [bitflip(gene, p) for gene in genome]

    return genome
