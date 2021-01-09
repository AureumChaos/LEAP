from collections import Counter

import pytest

from leap_ec import statistical_helpers as stat

##############################
# Tests for stochastic_equals()
##############################
def test_stochastic_equals1():
    """If the expected and observed dists are identical, return true."""
    observed = { 0: 1000, 1: 500 }
    expected = { 1: 500, 0: 1000 }
    assert(stat.stochastic_equals(expected, observed, p=0.001))


def test_stochastic_equals2():
    """Equal distributions should be equal, even if they only have 1 outcome."""
    observed = { 0: 1000 }
    expected = { 0: 1000 }
    assert(stat.stochastic_equals(expected, observed, p=0.001))


##############################
# Tests for equals_uniform()
##############################
def test_equals_uniform1():
    """If the observed dist is exactly uniform, return true."""
    observed = { 'A': 1000, 'B': 1000, 'C': 1000, 'D': 1000 }
    assert(stat.equals_uniform(observed, p=0.001))


def test_equals_uniform2():
    """If the observed dist is extremely non-uniform, return false."""
    observed = {'A': 15, 'B': 1000, 'C': 10555, 'D': 1 }
    assert(not stat.equals_uniform(observed, p=0.001))