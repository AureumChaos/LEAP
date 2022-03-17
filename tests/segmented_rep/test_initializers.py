"""Unit tests for initializers in the segmented representation package."""
import pytest
import random
import functools
from collections import Counter

from leap_ec import statistical_helpers as stat
from leap_ec.segmented_rep.initializers import create_segmented_sequence


test_sequence = [12345] # just an arbitrary sequence for testing

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


@pytest.mark.stochastic
def test_segmented_initializer_variable_length():
    """ created variable length segments

        We generate segments with length drawn from  U(1,5).  We would therefore
        expect the distribution of the number of segments and the distribution
        that was generated to be statistically significantly similar.
    """
    distribution_func = functools.partial(random.randint, a=1, b=5)

    segments = []
    segment_lengths = []

    N = 10000

    for i in range(N):
        # randomly generate a sequence of segments with the number of segments
        # drawn from a uniform distribution
        segments.append(create_segmented_sequence(distribution_func,
                                                  gen_sequence))

        # track the lengths of those segments
        segment_lengths.append(len(segments[-1]))

    distribution = Counter(segment_lengths)

    # TODO have a stat helper that can generate this conveniently
    # We expect the values to be evenly distrib in [1,5]
    expected_distribution = {1: N/5, 2: N/5, 3: N/5, 4: N/5, 5: N/5}

    assert stat.stochastic_equals(distribution, expected_distribution, p=0.001)
