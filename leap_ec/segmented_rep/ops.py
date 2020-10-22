#!/usr/bin/env python3
"""
    Segmented representation specific pipeline operators.
"""
from typing import Iterator, Callable
import random
from toolz import curry

from leap_ec.ops import compute_expected_probability, iteriter_op
from leap_ec.binary_rep.ops import flip

##############################
# Function segmented_bitflip
##############################
@curry
def segmented_bitflip(segment: list, mutation_prob: float = 1.0) -> list:
    """ Perform bitflip mutation on the given segment

    Intended to be used in conjunction with `apply_mutation()`

    TODO find a way to minimize copy and pasting from
    binary_rep.ops.mutate_bitflip.

    :param segment: to be mutated
    :type segment: list
    :param mutation_prob: actual probability of mutating a given segment element
    :type mutation_prob: float in [0.0,1.0]
    :return: mutated segment
    :rtype: list
    """
    actual_prob = compute_expected_probability(segment, expected_prob)

    mutated_segment = [flip(gene, actual_prob) for gene in segment]

    return mutated_segment


##############################
# Function apply_mutation
##############################
@curry
def apply_mutation(next_individual: Iterator,
                   mutator_func: Callable[[list, float], list],
                   expected_prob: float = 1.0) -> Iterator:
    """
    This expects next_individual to have a segmented representation; i.e.,
    a sequence of sequences.  `mutator_func` will be applied to each
    sub-sequence with the expected probability.  The expected probability
    applies to *all* the sequences, and defaults to a single mutation among
    all components, on average.

    :param next_individual: to mutation
    :type next_individual: Iterator
    :param mutator_func: function to be applied to each segment in the
        individual's genome; first argument is a segment, the second the
        expected probability of mutating each segment element.
    :param expected: expected mutations on average in [0.0,1.0]
    :type expected_prob: float
    :return: yielded mutated individual
    :rtype: Iterator
    """
    while True:
        individual = next(next_individual)

        # compute expected probability of mutation _per segment_
        per_segment_expected_prob = compute_expected_probability(expected_prob,
                                                                 individual.genome)

        # Apply mutation function using the expected probability to create a
        # new sequence of sequences to be assigned to the genome.
        mutated_genome = [mutator_func(segment, expected_prob=expected_prob)
                          for segment in individual.genome]

        individual.genome = mutated_genome

        yield individual