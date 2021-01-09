"""Unit tests for initializers in the integer representation package."""
from collections import Counter

import pytest

from leap_ec.int_rep.initializers import create_int_vector
from leap_ec import statistical_helpers as stat


@pytest.mark.stochastic
def test_create_int_vector():
    """Genomes created by this initializer should sample uniformly within each 
    gene's range."""
    N = 10000  # Well sample 10,000 independent genomes
    # Two genes, with two diffrent ranges
    init = create_int_vector(bounds=[ (0, 1), (55, 64) ])
    population = [ init() for _ in range(N) ]

    # Set up average distribution we expect to see for each gene,
    # as `value: expected_count` pairs
    # Both are uniform, but with different ranges.
    expected_dist0 = { 0: N/2, 1: N/2 }
    expected_dist1 = { 55: N/10, 56: N/10, 57: N/10, 58: N/10, 59: N/10, 
                       60: N/10, 61: N/10, 62: N/10, 63: N/10, 64: N/10 }
    
    # Count how many times we observe each value in the sampled genomes
    dist0 = Counter([ genome[0] for genome in population ])
    dist1 = Counter([ genome[1] for genome in population ])

    # Use a chi2 test to see if the observed gene-value counts are 
    # differ significantly from the expected distributions.
    p = 0.001
    assert(stat.stochastic_equals(expected_dist0, dist0, p=p))
    assert(stat.stochastic_equals(expected_dist1, dist1, p=p))