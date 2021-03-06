"""Unit tests for real-valued reproductive operators."""
import pytest
from scipy import stats

from leap_ec.individual import Individual
from leap_ec.real_rep import ops

##############################
# Tests for mutate_gaussian()
##############################
@pytest.mark.stochastic
def test_mutate_gaussian():
    """If we apply isotropic Gaussian mutation to a given genome a bunch of different times,
    the offsprings' genes should follow a Gaussian distribution around their parents' values."""
    N = 5000  # We'll sample 5,000 independent genomes

    gene0_values = []
    gene1_values = []

    for _ in range(N):
        # Set up two parents with fixed genomes, two genes each
        ind1 = Individual([0, 0.5])
        population = iter([ind1])
        
        # Mutate
        result = ops.mutate_gaussian(population, std=1.0, expected_num_mutations='isotropic')
        result = next(result)  # Pulse the iterator

        gene0_values.append(result.genome[0])
        gene1_values.append(result.genome[1])

    # Use a Kolmogorov-Smirnoff test to verify that the mutations follow a
    # Gaussian distribution with zero mean and unit variance
    p_threshold = 0.01

    # Gene 0 should follow N(0, 1.0)
    _, p = stats.kstest(gene0_values, 'norm')
    print(p)
    assert(p > p_threshold)

    # Gene 1 should follow N(0.5, 1.0)
    gene1_centered_values = [ x - 0.5 for x in gene1_values ]
    _, p = stats.kstest(gene1_centered_values, 'norm')
    print(p)
    assert(p > p_threshold)

    # Gene 1 should *not* follow N(0, 1.0)
    _, p = stats.kstest(gene1_values, 'norm')
    print(p)
    assert(p <= p_threshold)

    