"""Unit tests for operators in the integer representation package."""
from collections import Counter

import pytest
from scipy import stats
import toolz

from leap_ec.individual import Individual
import leap_ec.ops
import leap_ec.int_rep.ops as intrep_ops
from leap_ec import statistical_helpers as stat


##############################
# Tests for mutate_randint
##############################
def collect_two_gene_mutation_counts(mutator, N: int):
    """Helper to collect the distribution of results when we
    apply mutation to two small individuals."""
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
        result = mutator(population)
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

    return [ [ ind0_gene0_counts, ind0_gene1_counts ],
             [ ind1_gene0_counts, ind1_gene1_counts ] ]


@pytest.mark.stochastic
def test_mutate_randint1():
    """If you send me two individuals with two genes each and ask for 1 gene to
    be mutated on average, then on average each gene has a probability
    of 0.5 of being mutated."""

    N = 1000  # We'll sample 1,000 independent genomes
    mutator = intrep_ops.mutate_randint(bounds=[(0, 1), (0, 1)], expected_num_mutations=1)
    observed = collect_two_gene_mutation_counts(mutator, N)
    
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
    assert(stat.stochastic_equals(expected_ind0_gene0, observed[0][0], p=p))
    assert(stat.stochastic_equals(expected_ind0_gene1, observed[0][1], p=p))
    assert(stat.stochastic_equals(expected_ind1_gene0, observed[1][0], p=p))
    assert(stat.stochastic_equals(expected_ind1_gene1, observed[1][1], p=p))


@pytest.mark.stochastic
def test_mutate_randint2():
    """If we set the expected number of mutations to 2 when our genomes have
     only 2 genes, then each gene is always mutated, meaning individuals are
     completely resampled from a uniform distribution."""

    N = 1000  # We'll sample 1,000 independent genomes
    mutator = intrep_ops.mutate_randint(bounds=[(0, 1), (0, 1)], expected_num_mutations=2)
    observed = collect_two_gene_mutation_counts(mutator, N)

    # Expected distribution of mutations.
    # We arrive at this by the following reasoning: since we only have
    # two genes, our mutation probability is 2/L = 1.0.  So all four genes
    # should be sampled uniformly from the set {0, 1}.
    expected = { 0: 0.5*N, 1: 0.5*N }
    p = 0.001
    assert(stat.stochastic_equals(expected, observed[0][0], p=p))
    assert(stat.stochastic_equals(expected, observed[0][1], p=p))
    assert(stat.stochastic_equals(expected, observed[1][0], p=p))
    assert(stat.stochastic_equals(expected, observed[1][1], p=p))


@pytest.mark.stochastic
def test_mutate_randint3():
    """If you send me two individuals with two genes each and ask for a mutations
    probability of 0.2, then that's what will happen."""

    N = 1000  # We'll sample 1,000 independent genomes
    mutator = intrep_ops.mutate_randint(bounds=[(0, 1), (0, 1)], probability=0.2)
    observed = collect_two_gene_mutation_counts(mutator, N)
    
    # Expected distribution of mutations.
    # We arrive at this by the following reasoning: each gene has a 0.8
    # chance of not being mutated, in which case it keeps it original value.
    # Otherwise, it's value is sampled uniformly from the set {0, 1}.
    expected_ind0_gene0 = { 0: 0.8*N + 0.1*N, 1: 0.1*N }
    expected_ind0_gene1 = expected_ind0_gene0
    expected_ind1_gene0 = { 0: 0.1*N, 1: 0.8*N + 0.1*N }
    expected_ind1_gene1 = expected_ind1_gene0

    # Use a chi2 test to see if the observed gene-value counts are
    # differ significantly from the expected distributions.
    p = 0.001
    assert(stat.stochastic_equals(expected_ind0_gene0, observed[0][0], p=p))
    assert(stat.stochastic_equals(expected_ind0_gene1, observed[0][1], p=p))
    assert(stat.stochastic_equals(expected_ind1_gene0, observed[1][0], p=p))
    assert(stat.stochastic_equals(expected_ind1_gene1, observed[1][1], p=p))


@pytest.mark.stochastic
def test_mutate_randint4():
    """If you send me two individuals with two genes each and ask for a mutations
    probability of 1.0, then all genes should be completely resampled from a
    uniform distribution."""

    N = 1000  # We'll sample 1,000 independent genomes
    mutator = intrep_ops.mutate_randint(bounds=[(0, 1), (0, 1)], probability=1.0)
    observed = collect_two_gene_mutation_counts(mutator, N)
    
    # Expected distribution of mutations.
    # We arrive at this by the following reasoning: each gene has a 0.8
    # chance of not being mutated, in which case it keeps it original value.
    # Otherwise, it's value is sampled uniformly from the set {0, 1}.
    expected = { 0: 0.5*N, 1: 0.5*N }

    # Use a chi2 test to see if the observed gene-value counts are
    # differ significantly from the expected distributions.
    p = 0.001
    assert(stat.stochastic_equals(expected, observed[0][0], p=p))
    assert(stat.stochastic_equals(expected, observed[0][1], p=p))
    assert(stat.stochastic_equals(expected, observed[1][0], p=p))
    assert(stat.stochastic_equals(expected, observed[1][1], p=p))


def test_mutate_randint5():
    """If we fail to provide either expected_num_mutations or a probability parameter,
    an exception should occur when the operator is used."""

    mutator = intrep_ops.mutate_randint(bounds=[(0, 1), (0, 1)])
    ind1 = Individual([0, 0])
    ind2 = Individual([1, 1])
    population = iter([ind1, ind2])
    result = mutator(population)

    with pytest.raises(ValueError):
        # Pulse the iterator so mutation gets executed
        result = list(result)


def test_mutate_randint6():
    """If we provide a value for both expected_num_mutations and the probability parameter,
    an exception should occur when the operator is used."""

    mutator = intrep_ops.mutate_randint(bounds=[(0, 1), (0, 1)],
                                        expected_num_mutations=1,
                                        probability=0.1)
    ind1 = Individual([0, 0])
    ind2 = Individual([1, 1])
    population = iter([ind1, ind2])
    result = mutator(population)

    with pytest.raises(ValueError):
        # Pulse the iterator so mutation gets executed
        result = list(result)
    

def test_mutate_randint_pipe():
    """  This tests pipeline integration
    """
    ind1 = Individual([0, 0, 0])
    ind2 = Individual([1, 1, 1])
    population = iter([ind1, ind2])

    bounds = [(-100, 100), (0, 25), (-10, 10)]

    # Test that mutate_randint can be plugged into a pipeline since we
    # were experiencing an error when trying to do this.  The error turned out
    # to be that `bounds=` wasn't included in the call, which meant that python
    # tried to immediately invoke the `mutate_randint` instead of delaying
    # execution per the pipeline calls.
    results = toolz.pipe(population,
                         leap_ec.ops.clone,
                         intrep_ops.mutate_randint(bounds=bounds, expected_num_mutations=1),
                         # intrep_ops.mutate_randint(bounds), INCORRECT USAGE
                         leap_ec.ops.pool(size=2))

    assert len(results)


##############################
# Tests for mutate_binomial
##############################
def test_mutate_binomial_bounds():
    """If we apply a wide mutation distribution repeatedly, it should never stray
    outside of the provided bounds.
    
    This test runs the stochastic function repeatedly, but we don't mark it as a 
    stochastic test because it's and should never fail unless there is actually a
    fault."""
    operator = intrep_ops.mutate_binomial(std=20, bounds=[(0, 10), (2, 20)],
                                          expected_num_mutations=1)

    N = 100
    for i in range(N):
        population = iter([ Individual([5,10]) ])
        mutated = next(operator(population))
        assert(mutated.genome[0] >= 0)
        assert(mutated.genome[0] <= 10)
        assert(mutated.genome[1] >= 2)
        assert(mutated.genome[1] <= 20)


@pytest.mark.stochastic
def test_mutate_binomial_dist():
    """When we apply binomial mutation repeatedly, the resulting distribution
    of offspring should follow the expected theoretical distribution."""

    N = 5000  # Number of mutantes to generate
    binom_n = 10000  # "coin flips" parameter for the binomial
    std = 2.5  # Standard deviation of the mutation distribution

    # We'll set up our operator with infinite bounds, so we needn't worry about clipping
    operator = intrep_ops.mutate_binomial(std=std, expected_num_mutations=2,
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
    binom_p = intrep_ops._binomial_p_from_std(binom_n, std)
    binom = stats.binom(binom_n, binom_p)
    mu = binom_n * binom_p  # Mean of a binomial distribution is n*p

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

def test_mutate_binomial_err1():
    """If we fail to provide either expected_num_mutations or a probability parameter,
    an exception should occur when the operator is used."""

    mutator = intrep_ops.mutate_binomial(std=1, bounds=[(0, 1), (0, 1)])
    ind1 = Individual([0, 0])
    ind2 = Individual([1, 1])
    population = iter([ind1, ind2])
    result = mutator(population)

    with pytest.raises(ValueError):
        # Pulse the iterator so mutation gets executed
        result = list(result)


def test_mutate_binomial_err2():
    """If we provide a value for both expected_num_mutations and the probability parameter,
    an exception should occur when the operator is used."""

    mutator = intrep_ops.mutate_binomial(std=1, bounds=[(0, 1), (0, 1)],
                                        expected_num_mutations=1,
                                        probability=0.1)
    ind1 = Individual([0, 0])
    ind2 = Individual([1, 1])
    population = iter([ind1, ind2])
    result = mutator(population)

    with pytest.raises(ValueError):
        # Pulse the iterator so mutation gets executed
        result = list(result)