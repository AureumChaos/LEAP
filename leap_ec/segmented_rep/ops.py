#!/usr/bin/env python3
"""
    Segmented representation specific pipeline operators.
"""
from typing import Iterator, Callable
import random
from toolz import curry

from leap_ec.ops import compute_expected_probability, iteriter_op


##############################
# Function apply_mutation
##############################
@curry
@iteriter_op
def apply_mutation(next_individual: Iterator,
                   mutator: Callable[[list, float], list],
                   expected_num_mutation: float = 1.0) -> Iterator:
    """
    This expects next_individual to have a segmented representation; i.e.,
    a sequence of sequences.  `mutator_func` will be applied to each
    sub-sequence with the expected probability.  The expected probability
    applies to *all* the sequences, and defaults to a single mutation among
    all components, on average.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.binary_rep.ops import perform_mutate_bitflip
    >>> original = Individual([[0,0],[1,1]])
    >>> mutated = next(apply_mutation(iter([original]),mutator=perform_mutate_bitflip))

    :param next_individual: to mutation
    :param mutator: function to be applied to each segment in the
        individual's genome; first argument is a segment, the second the
        expected probability of mutating each segment element.
    :param expected: expected mutations on average in [0.0,1.0]
    :return: yielded mutated individual
    """
    while True:
        individual = next(next_individual)

        # compute expected probability of mutation _per segment_
        per_segment_expected_prob = compute_expected_probability(
            expected_num_mutation, individual.genome)

        # Apply mutation function using the expected probability to create a
        # new sequence of sequences to be assigned to the genome.
        mutated_genome = [mutator(segment, expected_num_mutations=per_segment_expected_prob)
                          for segment in individual.genome]

        individual.genome = mutated_genome

        yield individual