"""Unit tests for operators in the integer representation package."""
from collections import Counter

import pytest
from scipy import stats

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
    p = 0.001
    assert(stat.stochastic_equals(expected_ind0_gene0, ind0_gene0_counts, p=p))
    assert(stat.stochastic_equals(expected_ind0_gene1, ind0_gene1_counts, p=p))
    assert(stat.stochastic_equals(expected_ind1_gene0, ind1_gene0_counts, p=p))
    assert(stat.stochastic_equals(expected_ind1_gene1, ind1_gene1_counts, p=p))


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
    p = 0.001
    assert(stat.stochastic_equals(expected, ind0_gene0_counts, p=p))
    assert(stat.stochastic_equals(expected, ind0_gene1_counts, p=p))
    assert(stat.stochastic_equals(expected, ind1_gene0_counts, p=p))
    assert(stat.stochastic_equals(expected, ind1_gene1_counts, p=p))


##############################
# Tests for mutate_binomial
##############################
def test_binomial_bounds():
    """If we apply a wide mutation distribution repeatedly, it should never stray
    outside of the provided bounds.
    
    This test runs the stochastic function repeatedly, but we don't mark it as a 
    stochastic test because it's and should never fail unless there is actually a
    fault."""
    operator = ops.mutate_binomial(std=20, bounds=[(0, 10), (2, 20)])

    N = 100
    for i in range(N):
        population = iter([ Individual([5,10]) ])
        mutated = next(operator(population))
        assert(mutated.genome[0] >= 0)
        assert(mutated.genome[0] <= 10)
        assert(mutated.genome[1] >= 2)
        assert(mutated.genome[1] <= 20)


@pytest.mark.stochastic
def test_binomial_dist():
    """When we apply binomial mutation repeatedly, the resulting distribution
    of offspring should follow the expected theoretical distribution."""

    N = 5000  # Number of mutantes to generate
    binom_n = 10000  # "coin flips" parameter for the binomial
    std = 2.5  # Standard deviation of the mutation distribution

    # We'll set up our operator with infinite bounds, so we needn't worry about clipping
    operator = ops.mutate_binomial(std=std, expected_num_mutations=2,
                                   bounds=[(-float('inf'), float('inf')), (-float('inf'), float('inf'))])

    # Any value could appear, but we'll focus on measuring just a few
    # nearby values
    genome = [5, 10]
    gene0_observed_dist = { '3': 0, '4': 0, '5': 0, '6': 0, '7':0 }
    gene1_observed_dist = { '8': 0, '9': 0, '10': 0, '11': 0, '12': 0 }

    # Count the observed mutations in N trials
    for i in range(N):
        population = iter([ Individual(genome) ])
        mutated = next(operator(population))
        gene0, gene1 = mutated.genome
        gene0, gene1 = str(gene0), str(gene1)

        # Count the observed values of the first gene
        if gene0 in gene0_observed_dist.keys():
            gene0_observed_dist[gene0] += 1

        # Count the observed values of the second gene
        if gene1 in gene1_observed_dist.keys():
            gene1_observed_dist[gene1] += 1

    # Set up the expected distribution by using SciPy's binomial PMF function
    binom_p = ops._binomial_p_from_std(binom_n, std)
    binom = stats.binom(binom_n, binom_p)
    mu = binom_n * binom_p  # Mean of a binomial distribution is n*p
    for k in gene0_observed_dist.keys():
        print(f"k: {int(k)}, arg: {mu - (genome[0] - int(k))}, pmf: {binom.pmf(mu - (genome[0] - int(k)))}")

    gene0_expected_dist = { k: int(N*binom.pmf(int(mu - (genome[0] - int(k))))) for k in gene0_observed_dist.keys() }
    gene1_expected_dist = { k: int(N*binom.pmf(int(mu - (genome[1] - int(k))))) for k in gene1_observed_dist.keys() }

    # Toss all the other values under one value
    gene0_observed_dist['other'] = N - sum(gene0_observed_dist.values())
    gene1_observed_dist['other'] = N - sum(gene1_observed_dist.values())
    gene0_expected_dist['other'] = N - sum(gene0_expected_dist.values())
    gene1_expected_dist['other'] = N - sum(gene1_expected_dist.values())

    p = 0.01
    assert(stat.stochastic_equals(gene0_expected_dist, gene0_observed_dist, p=p))
    assert(stat.stochastic_equals(gene1_expected_dist, gene1_observed_dist, p=p))
