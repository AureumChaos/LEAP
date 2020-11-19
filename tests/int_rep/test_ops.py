"""Unit tests for operators in the integer representation package."""
from collections import Counter

import pytest

from leap_ec.individual import Individual
import leap_ec.int_rep.ops as ops
from leap_ec import statistical_helpers as stat


##############################
# Tests for mutate_randint
##############################
@pytest.mark.stochastic
def test_mutate_randint1():
    """If you send me two individuals with two genes each and keep the 
    default mutation rate, then on average, each gene has a probability 
    of 0.5 of being mutated."""

    N = 1000

    ind0_gene0_values = []
    ind0_gene1_values = []
    ind1_gene0_values = []
    ind1_gene1_values = []

    for _ in range(N):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        population = iter([ind1, ind2])

        result = ops.mutate_randint(population, bounds=[(0, 1), (0, 1)])
        result = list(result)  # Pulse the iterator

        ind0_gene0_values.append(result[0].genome[0])
        ind0_gene1_values.append(result[0].genome[1])
        ind1_gene0_values.append(result[1].genome[0])
        ind1_gene1_values.append(result[1].genome[1])

    ind0_gene0_counts = Counter(ind0_gene0_values)
    ind0_gene1_counts = Counter(ind0_gene1_values)
    ind1_gene0_counts = Counter(ind1_gene0_values)
    ind1_gene1_counts = Counter(ind1_gene1_values)

    # Expected distribution of mutations: 0.5 chance of not being mutated (keep original value), else uniform
    expected_ind0_gene0 = { 0: 0.5*N + 0.25*N, 1: 0.25*N }
    expected_ind0_gene1 = expected_ind0_gene0
    expected_ind1_gene0 = { 0: 0.25*N, 1: 0.5*N + 0.25*N }
    expected_ind1_gene1 = expected_ind1_gene0

    assert(stat.stochastic_equals(expected_ind0_gene0, ind0_gene0_counts))
    assert(stat.stochastic_equals(expected_ind0_gene1, ind0_gene1_counts))
    assert(stat.stochastic_equals(expected_ind1_gene0, ind1_gene0_counts))
    assert(stat.stochastic_equals(expected_ind1_gene1, ind1_gene1_counts))


@pytest.mark.stochastic
def test_mutate_randint2():
    """If we set the expected number of mutations to 2 when our genomes have
     only 2 genes, then each gene is always mutated, meaning individuals are
     completely resampled from a uniform distribution."""

    N = 1000

    ind0_gene0_values = []
    ind0_gene1_values = []
    ind1_gene0_values = []
    ind1_gene1_values = []

    for _ in range(N):
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        population = iter([ind1, ind2])

        result = ops.mutate_randint(population, bounds=[(0, 1), (0, 1)], expected_num_mutations=2)
        result = list(result)  # Pulse the iterator

        ind0_gene0_values.append(result[0].genome[0])
        ind0_gene1_values.append(result[0].genome[1])
        ind1_gene0_values.append(result[1].genome[0])
        ind1_gene1_values.append(result[1].genome[1])

    ind0_gene0_counts = Counter(ind0_gene0_values)
    ind0_gene1_counts = Counter(ind0_gene1_values)
    ind1_gene0_counts = Counter(ind1_gene0_values)
    ind1_gene1_counts = Counter(ind1_gene1_values)

    expected = { 0: 0.5*N, 1: 0.5*N }

    assert(stat.stochastic_equals(expected, ind0_gene0_counts))
    assert(stat.stochastic_equals(expected, ind0_gene1_counts))
    assert(stat.stochastic_equals(expected, ind1_gene0_counts))
    assert(stat.stochastic_equals(expected, ind1_gene1_counts))
