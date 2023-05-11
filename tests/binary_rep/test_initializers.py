"""Unit test for binary-representation initialization functions."""
import numpy as np
import pytest

from leap_ec import Individual
from leap_ec.binary_rep import initializers
import leap_ec.statistical_helpers as stat


##############################
# Tests for create_binary_sequence()
##############################
@pytest.mark.stochastic
def test_create_binary_sequence():
    """Calling create_binary_sequence() gives us a function that returns vectors whose elements
    are True or False with equal probability."""
    N = 5000
    observed_dist = {True: 0, False: 0 }

    # Sample N individuals of length 10
    initialize = initializers.create_binary_sequence(length=10)
    individuals = [ initialize() for _ in range(N) ]

    # Count the number of times True an False, respectively, appear as gene values
    values, counts = np.unique(individuals, return_counts=True)
    observed_dist = dict(zip(values, counts))
    assert(N*10 == sum(observed_dist.values()))

    # We expect a 50/50 split between True and False
    expected_dist = {
        True: N*10/2,
        False: N*10/2
    }

    # Use a χ-squared test to see if our experiment matches what we expect
    p = 0.01
    assert(stat.stochastic_equals(expected_dist, observed_dist, p=p))