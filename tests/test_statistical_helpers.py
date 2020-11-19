from collections import Counter

import pytest

from leap_ec import statistical_helpers as stat

##############################
# Tests for stochastic_equals()
##############################
@pytest.mark.stochastic
def test_stochastic_equals1():
    """If the expected and observed dists are identical, return true."""
    observed = { 0: 1000, 1: 500 }
    expected = { 1: 500, 0: 1000 }
    assert(stat.stochastic_equals(expected, observed))


@pytest.mark.stochastic
def test_stochastic_equals2():
    """Equal distributions should be equal, even if they only have 1 outcome."""
    observed = { 0: 1000 }
    expected = { 0: 1000 }
    assert(stat.stochastic_equals(expected, observed))