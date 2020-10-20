#!/usr/bin/env python3
"""
    Segmented representation specific pipeline operators.
"""
from typing import Iterator
import random
from toolz import curry

from leap_ec.ops import compute_expected_probability, iteriter_op


@curry
def apply_mutation(next_individual: Iterator, mutator_func, expected: float = 1) -> Iterator:
    """
    This expects next_individual to have a segmented representation; i.e.,
    a sequence of sequences.  `mutator_func` will be applied to each
    sub-sequence with the expected probability.  The expected probability
    applies to *all* the sequences, and defaults to a single mutation among
    all components, on average.

    :param next_individual: to mutation
    :type next_individual: Iterator
    :param mutator_func: function to be applied to
    :param expected: expected mutations on average
    :type expected: float
    :return: yielded mutated individual
    :rtype: Iterator
    """

    while True:
        individual = next(next_individual)

        yield individual