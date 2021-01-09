"""Helpers for testing the output of stochastic functions."""
from typing import Dict

from scipy.stats import chisquare


def collect_distribution(function, samples: int):
    """Count the number of times the given function returns each
    output value."""
    assert(callable(function))

    outputs = {}
    for i in range(samples):
        o = function()
        outputs[o] = outputs.get(o, 0) + 1
    
    return outputs


def _normalize_dicts(dict1, dict2):
    """Convert two dicts to lists that are aligned to each other."""

    def add_keys_from(dist1, dist2):
        """If dist1 contains a key that dist2 doesn't, add it to dict2."""
        for k in dist1.keys():
            if k not in dist2:
                dist2[k] = 0

    def values_sorted_by_key(dist):
        """Get the values of dist, sorted by the keys."""
        return [dist[k] for k in sorted(dist.keys())]
    
    add_keys_from(dict1, dict2)
    add_keys_from(dict2, dict1)

    n_dict1 = values_sorted_by_key(dict1)
    n_dict2 = values_sorted_by_key(dict2)

    return n_dict1, n_dict2


def stochastic_chisquare(expected_distribution, distribution):
    """Use a $\\chi^2$ distribution to compute a p-value for the probability of
    rejecting the hypothesis that the given distribution matches the expected
    distribution.
    
    This takes two dictionaries of values:

    >>> expected_distribution = { 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10}
    >>> distribution = { 1: 5, 2: 8, 3: 9, 4: 8, 5: 10, 6: 20}
    >>> stochastic_chisquare(expected_distribution, distribution)
    0.01990...
    """
    assert(sum(expected_distribution.values()) == sum(distribution.values())), f"The distributions have {sum(expected_distribution.values())} and {sum(distribution.values())} samples, respectively, but must be equal."
    
    values, expected_values = _normalize_dicts(distribution, expected_distribution)

    _, p_value = chisquare(values, expected_values)
    return p_value


def stochastic_equals(expected_distribution: Dict, observed_distribution: Dict, p: float) -> bool:
    """Use a $\\chi^2$ test to determine whether two discrete distributions are
    equal.

    For example, we do not reject the hypothesis that `[5060, 4940]` comes from a uniform
    distribution:

    >>> expected = { 0: 5000, 1: 5000 }
    >>> observed = { 0: 5060, 1: 4940 }
    >>> stochastic_equals(expected, observed, p=0.01)
    True

    Here we also do not reject the hypothesis that a 6-sided die is unbiased:
    
    >>> expected = { 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10}
    >>> observed = { 1: 5, 2: 8, 3: 9, 4: 8, 5: 10, 6: 20}
    >>> stochastic_equals(expected, observed, p=0.01)
    True

    But we would have if we used a 95% significance level instead of 99%:
    >>> stochastic_equals(expected, observed, p=0.05)
    False
    
    """

    # Handle the special case where each variable has only one possible value
    values, expected_values = _normalize_dicts(observed_distribution, expected_distribution)
    if len(values) == 1:
        return values == expected_values

    # Otherwise, do a chi2 test
    p_value = stochastic_chisquare(expected_distribution, observed_distribution)
    return p_value > p


def equals_uniform(observed_distribution: Dict, p: float) -> bool:
    """Use a $\\chi^2$ test to determine whether the observed distribution is uniform.
    
    This offers convenience over stochastic_equals(), because the expected distribution
    doesn't have to be manually specified.

    For example, we do not reject the hypothesis that `[5060, 4940]` comes from a uniform
    distribution:

    >>> observed = { 0: 5060, 1: 4940 }
    >>> equals_uniform(observed, p=0.01)
    True

    The keys are arbitrary, so we can use them to clearly express what we are testing:

    >>> observed = { 'Left': 101, 'Right': 100, 'Up': 99, 'Down': 100 }
    >>> equals_uniform(observed, p=0.01)
    True

    """
    n = sum( v for k, v in observed_distribution.items() )
    num_keys = len(observed_distribution.keys())
    expected_distribution = { k: n/num_keys for k in observed_distribution.keys() }
    return stochastic_equals(expected_distribution, observed_distribution, p)
