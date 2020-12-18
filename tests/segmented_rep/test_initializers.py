"""Unit tests for initializers in the segmented representation package."""
import pytest

from leap_ec.segmented_rep.initializers import create_segmented_sequence

test_sequence = [1111]

def gen_sequence():
    """ return an arbitrary static test_sequence """
    return test_sequence

def test_segmented_initializer_fixed_length():
    """ created fixed length segments """

    segments = create_segmented_sequence(1, gen_sequence)
    assert segments == [test_sequence]

    segments = create_segmented_sequence(2, gen_sequence)
    assert segments == [test_sequence, test_sequence]

    segments = create_segmented_sequence(3, gen_sequence)
    assert segments == [test_sequence, test_sequence, test_sequence]
