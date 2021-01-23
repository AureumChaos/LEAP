"""Unit tests for pipeline operators in the segmented representation package."""
import pytest
import random
import functools
from collections import Counter

from leap_ec.individual import Individual

from leap_ec import statistical_helpers as stat
from leap_ec.ops import n_ary_crossover
from leap_ec.segmented_rep.initializers import create_segmented_sequence
from leap_ec.segmented_rep.ops import remove_segment, add_segment, copy_segment

test_sequence = [12345]  # just an arbitrary sequence for testing

def gen_sequence():
    """ return an arbitrary static test_sequence """
    return test_sequence


def test_segmented_remove():
    original = Individual([[0, 0], [1, 1]])
    mutated = next(remove_segment(iter([original]), probability=1.0))
    assert mutated.genome == [[0, 0]] or mutated.genome == [[1, 1]]


def test_segmented_add():
    # Test with append first
    original = Individual([[0, 0], [1, 1]])
    mutated = next(add_segment(iter([original]),
                               seq_initializer=gen_sequence,
                               probability=1.0,
                               append=True))
    assert mutated.genome == [[0, 0], [1, 1], test_sequence]


    # Test without append, so segment can inserted in one of three locations
    possible_outcomes = [[test_sequence, [0, 0], [1, 1]],
                         [[0, 0], test_sequence, [1, 1]],
                         [[0, 0], [1, 1], test_sequence]]

    for i in range(20):
        original = Individual([[0, 0], [1, 1]])
        mutated = next(add_segment(iter([original]),
                                   seq_initializer=gen_sequence,
                                   probability=1.0,
                                   append=False))
        assert mutated.genome in possible_outcomes


def test_segmented_copy():
    original = Individual([[0, 0], [1, 1]])
    mutated = next(copy_segment(iter([original]),
                                probability=1.0,
                                append=True))

    possible_outcomes = [[[0, 0], [1, 1], [0, 0]],
                         [[0, 0], [1, 1], [1, 1]],
                         ]

    assert mutated.genome in possible_outcomes

    possible_outcomes = [[[0, 0], [0, 0], [1, 1]],
                         [[0, 0], [1, 1], [0, 0]],
                         [[1, 1], [0, 0], [1, 1]],
                         [[0, 0], [1, 1], [1, 1]],
                         ]

    # TODO: it would be better to build a histogram of expected outcomes and
    # do the chi square test
    for i in range(20):
        original = Individual([[0, 0], [1, 1]])
        mutated = next(copy_segment(iter([original]),
                                    probability=1.0,
                                    append=False))

        assert mutated.genome in possible_outcomes


def test_segmented_crossover():
    """ test that k-ary crossover works as expected for fixed and variable
        length segments
    """
    a = Individual([[0, 0], [1, 1]])
    b = Individual([[1, 1], [0, 0]])

    result = n_ary_crossover(iter([a,b]))
    c = next(result)
    d = next(result)

    possible_outcomes = [[[0, 0], [1, 1]],
                         [[1, 1], [0, 0]],
                         [[0, 0], [0, 0]],
                         [[1, 1], [1, 1]]]

    assert c.genome in possible_outcomes and d.genome in possible_outcomes

    # Now for genomes of different lengths
    # TODO I need to *carefully* review the possible crossover possibilities
    possible_outcomes = [[],
                         [[0, 0]],
                         [[1, 1]],
                         [[2, 2]],
                         [[0, 0], [1, 1]],
                         [[1, 1], [2, 2]],
                         [[2, 2], [1, 1]],
                         [[2, 2], [0, 0], [1, 1]],
                         [[0, 0], [2, 2], [1, 1]],
                         [[0, 0], [1, 1], [2, 2]],
                         ]

    for _ in range(20):
        a = Individual([[0, 0], [1, 1]])
        b = Individual([[2, 2]])

        result = n_ary_crossover(iter([a, b]), num_points=1)
        c = next(result)
        d = next(result)

        assert c.genome in possible_outcomes
        assert d.genome in possible_outcomes
