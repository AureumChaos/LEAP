"""
    Unit tests for binary-representation operators.
"""
import itertools

import numpy as np
import pytest

from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.individual import Individual
import leap_ec.binary_rep.ops as ops
import leap_ec.statistical_helpers as stat


##############################
# Tests for mutate_bitflip
##############################
def test_mutate_bitflip():
    """If we mutated a 2-bit genome with expected_num_mutations=2,
    each bit has a 100% mutation probability and will be flipped."""
    # Create a very simple individual with two binary genes of all ones.
    ind = [Individual(np.array([1, 1]),
                      problem=MaxOnes())]

    # Now mutate the individual such that we *expect both bits to bitflip*
    mutated_ind = next(ops.mutate_bitflip(iter(ind), expected_num_mutations=2))

    assert np.all(mutated_ind.genome == [0, 0])

    # Of course, since we didn't clone the original, well, that actually got
    # zapped, too.

    assert np.all(ind[0].genome == [0, 0])


@pytest.mark.stochastic
def test_mutate_bitflip1():
    """If we set expected_num_mutations=1, then each bit should be flipped with probability of
    1/length on average."""

    N = 5000  # Number of sample populations to mutate
    # Prepare to count how many times each locus is and is not mutated
    observed_dist = {'0_flipped': 0, '1_flipped': 0, '2_flipped': 0, '3_flipped': 0,
                     '0_not_flipped': 0, '1_not_flipped': 0, '2_not_flipped': 0, '3_not_flipped': 0 }

    # Run crossover N times on a fixed pair of two-gene individuals
    for i in range(N):

        # A simple test population
        pop = [Individual(np.array([0, 0, 0, 0])),
               Individual(np.array([1, 1, 1, 1]))]
        # Mutate both individuals
        new_pop = list(itertools.islice(ops.mutate_bitflip(iter(pop), expected_num_mutations=1), 2))

        # Check which bits of the 0th individual were changed
        for i, v in enumerate(new_pop[0].genome):
            if v == 1:
                observed_dist[f"{i}_flipped"] += 1
            else:
                observed_dist[f"{i}_not_flipped"] += 1

        # Check which bits of the 1st individual were changed
        for i, v in enumerate(new_pop[1].genome):
            if v == 0:
                observed_dist[f"{i}_flipped"] += 1
            else:
                observed_dist[f"{i}_not_flipped"] += 1

    # The average counts we expect to see of each event
    expected_dist = {
        '0_flipped': N*2*1./4,  # each of the N*2 individuals we mutated should have its 0th bit flipped 1/4 of the time
        '1_flipped': N*2*1./4,
        '2_flipped': N*2*1./4,
        '3_flipped': N*2*1./4,
        '0_not_flipped': N*2*3./4,  # each of the N*2 individuals we mutated should have its 0th bit NOT flipped 3/4 of the time
        '1_not_flipped': N*2*3./4,
        '2_not_flipped': N*2*3./4,
        '3_not_flipped': N*2*3./4
    }

    # Use a χ-squared test to see if our experiment matches what we expect
    p = 0.01
    assert(stat.stochastic_equals(expected_dist, observed_dist, p=p))