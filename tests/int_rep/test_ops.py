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

    N = 1000  # We'll sample 1,000 independent genomes

    # Set up arrays to collect the values of 4 different loci after mutation
    ind0_gene0_values = []
    ind0_gene1_values = []
    ind1_gene0_values = []
    ind1_gene1_values = []

    for _ in range(N):
        # Set up two parents with fixed genomes, two genes each
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        population = iter([ind1, ind2])

        # Mutate the parents
        result = ops.mutate_randint(population, bounds=[(0, 1), (0, 1)])
        result = list(result)  # Pulse the iterator

        # Collect the values of each of the genes after mutation
        ind0_gene0_values.append(result[0].genome[0])
        ind0_gene1_values.append(result[0].genome[1])
        ind1_gene0_values.append(result[1].genome[0])
        ind1_gene1_values.append(result[1].genome[1])

    # Count the number of times that each gene value occurs at each locus
    ind0_gene0_counts = Counter(ind0_gene0_values)
    ind0_gene1_counts = Counter(ind0_gene1_values)
    ind1_gene0_counts = Counter(ind1_gene0_values)
    ind1_gene1_counts = Counter(ind1_gene1_values)

    # Expected distribution of mutations.
    # We arrive at this by the following reasoning: each gene has a 1/L = 0.5
    # chance of not being mutated, in which case it keeps it original value.
    # Otherwise, it's value is sampled uniformly from the set {0, 1}.
    expected_ind0_gene0 = { 0: 0.5*N + 0.25*N, 1: 0.25*N }
    expected_ind0_gene1 = expected_ind0_gene0
    expected_ind1_gene0 = { 0: 0.25*N, 1: 0.5*N + 0.25*N }
    expected_ind1_gene1 = expected_ind1_gene0

    # Use a chi2 test to see if the observed gene-value counts are 
    # differ significantly from the expected distributions.
    assert(stat.stochastic_equals(expected_ind0_gene0, ind0_gene0_counts))
    assert(stat.stochastic_equals(expected_ind0_gene1, ind0_gene1_counts))
    assert(stat.stochastic_equals(expected_ind1_gene0, ind1_gene0_counts))
    assert(stat.stochastic_equals(expected_ind1_gene1, ind1_gene1_counts))


@pytest.mark.stochastic
def test_mutate_randint2():
    """If we set the expected number of mutations to 2 when our genomes have
     only 2 genes, then each gene is always mutated, meaning individuals are
     completely resampled from a uniform distribution."""

    N = 1000  # We'll sample 1,000 independent genomes

    # Set up arrays to collect the values of 4 different loci after mutation
    ind0_gene0_values = []
    ind0_gene1_values = []
    ind1_gene0_values = []
    ind1_gene1_values = []

    for _ in range(N):
        # Set up two parents with fixed genomes, two genes each
        ind1 = Individual([0, 0])
        ind2 = Individual([1, 1])
        population = iter([ind1, ind2])

        # Mutate the parents
        result = ops.mutate_randint(population, bounds=[(0, 1), (0, 1)], expected_num_mutations=2)
        result = list(result)  # Pulse the iterator

        # Collect the values of each of the genes after mutation
        ind0_gene0_values.append(result[0].genome[0])
        ind0_gene1_values.append(result[0].genome[1])
        ind1_gene0_values.append(result[1].genome[0])
        ind1_gene1_values.append(result[1].genome[1])

    # Count the number of times that each gene value occurs at each locus
    ind0_gene0_counts = Counter(ind0_gene0_values)
    ind0_gene1_counts = Counter(ind0_gene1_values)
    ind1_gene0_counts = Counter(ind1_gene0_values)
    ind1_gene1_counts = Counter(ind1_gene1_values)


    # Expected distribution of mutations.
    # We arrive at this by the following reasoning: since we only have
    # two genes, our mutation probability is 2/L = 1.0.  So all four genes
    # should be sampled uniformly from the set {0, 1}.
    expected = { 0: 0.5*N, 1: 0.5*N }

    assert(stat.stochastic_equals(expected, ind0_gene0_counts))
    assert(stat.stochastic_equals(expected, ind0_gene1_counts))
    assert(stat.stochastic_equals(expected, ind1_gene0_counts))
    assert(stat.stochastic_equals(expected, ind1_gene1_counts))
