"""Unit tests for initializers in the segmented representation package."""
import pytest

from leap_ec.segmented_rep.initializers import create_segmented_sequence


def test_segmented_initializer_fixed_length():
    """ created fixed length segments """
    sequence = [1111]
    def test_sequence():
        """ return an arbitrary static sequence """
        return sequence

    segments = create_segmented_sequence(1, test_sequence)
    assert segments == [sequence]

    segments = create_segmented_sequence(2, test_sequence)
    assert segments == [sequence, sequence]

    segments = create_segmented_sequence(3, test_sequence)
    assert segments == [sequence, sequence, sequence]
