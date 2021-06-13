"""Unit tests for pipeline operators in the segmented representation package."""
import pytest
import random
import functools
from collections import Counter

from leap_ec.individual import Individual

from leap_ec import statistical_helpers as stat
from leap_ec.ops import n_ary_crossover
from leap_ec.binary_rep.ops import genome_mutate_bitflip
from leap_ec.segmented_rep.initializers import create_segmented_sequence
from leap_ec.segmented_rep.ops import apply_mutation, remove_segment, add_segment, copy_segment


##############################
# Test Fixtures
##############################
test_sequence = [12345]  # just an arbitrary sequence for testing

@pytest.fixture
def gen_sequence():
    """Return a function that returns an arbitrary static test_sequence."""
    def f():
        return test_sequence
    return f


##############################
# Tests for apply_mutation()
##############################
def test_apply_mutation():
    """Applying segment-wise mutation operators with expected_num_mutations=len(genome) should
    result in every gene of every segment being mutated."""
    mutation_op = apply_mutation(mutator=genome_mutate_bitflip, expected_num_mutations=4)
    original = Individual([[0,0],[1,1]])
    mutated = next(mutation_op(iter([original])))
    assert(mutated.genome == [[1, 1], [0, 0]])


##############################
# Tests for remove_segment()
##############################
def test_segmented_remove():
    original = Individual([[0, 0], [1, 1]])
    mutated = next(remove_segment(iter([original]), probability=1.0))
    assert mutated.genome == [[0, 0]] or mutated.genome == [[1, 1]]


##############################
# Tests for add_segment()
##############################
def test_segmented_add(gen_sequence):
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


##############################
# Tests for copy_segment()
##############################
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


##############################
# Tests for n_ary_crossover() on segmented genomes
##############################
def test_segmented_crossover():
    """ test that n-ary crossover works as expected for fixed and variable
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
