"""Unit tests for initializers in the integer representation package."""
from collections import Counter

import pytest

from leap_ec.int_rep.initializers import create_int_vector
from leap_ec import statistical_helpers as stat


@pytest.mark.stochastic
def test_create_int_vector():
    """Genomes created by this initializer should sample uniformly within each 
    gene's range."""
    N = 10000
    init = create_int_vector(bounds=[ (0, 1), (55, 64) ])
    population = [ init() for _ in range(N) ]

    expected_dist0 = { 0: N/2, 1: N/2 }
    expected_dist1 = { 55: N/10, 56: N/10, 57: N/10, 58: N/10, 59: N/10, 
                       60: N/10, 61: N/10, 62: N/10, 63: N/10, 64: N/10 }
    
    dist0 = Counter([ genome[0] for genome in population ])
    dist1 = Counter([ genome[1] for genome in population ])

    assert(stat.stochastic_equals(expected_dist0, dist0))
    assert(stat.stochastic_equals(expected_dist1, dist1))