import random
from typing import Iterator

import numpy as np
from toolz import curry

from leap_ec.ops import compute_expected_probability, iteriter_op


##############################
# Function mutate_randint
##############################
@curry
@iteriter_op
def mutate_randint(next_individual: Iterator, bounds,
                   expected_num_mutations = None,
                   probability = None
                   ) -> Iterator:
    """Perform randint mutation on each individual in an iterator (population).

    This operator replaces randomly selected genes with an integer samples
    from a uniform distribution.

    :param bounds: test_sequence of bounds tuples; e.g., [(1,2),(3,4)]
    :param expected_num_mutations: on average how many mutations done (specificy either this or probability, but not both)
    :param probability: the probability of mutating any given gene (specificy either this or expected_num_mutations, but not both)

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.int_rep.ops import mutate_randint

    >>> population = iter([ Individual([1,1]) ])
    >>> operator = mutate_randint(expected_num_mutations=1, bounds=[(0, 10), (0, 10)])
    >>> mutated = next(operator(population))
    """
    if (expected_num_mutations is not None) and (probability is not None):
        raise ValueError("Received parameters for 'probability' and 'expected_num_mutations', but can only use one or the other.")
    if (expected_num_mutations is None) and (probability is None):
        raise ValueError("Received no value for 'probability' or 'expected_num_mutations'.  Must have one.")
    if (probability is not None) and ((probability < 0) or (probability > 1)):
        raise ValueError(f"The value of 'probability' is {probability}, but must be >= 0 and <= 1.")

    while True:
        try:
            individual = next(next_individual)
        except StopIteration:
            return

        individual.genome = individual_mutate_randint(individual.genome, bounds,
                                                      expected_num_mutations=expected_num_mutations,
                                                      probability=probability)

        individual.fitness = None  # invalidate fitness since we have new genome

        yield individual


##############################
# Function individual_mutate_randint
##############################
@curry
def individual_mutate_randint(genome: list,
                              bounds: list,
                              expected_num_mutations = None,
                              probability = None) -> list:
    """ Perform random-integer mutation on a particular genome.

        >>> genome = [42, 12]
        >>> bounds = [(0,50), (-10,20)]
        >>> new_genome = individual_mutate_randint(genome, bounds, expected_num_mutations=1)

        :param genome: test_sequence of integers to be mutated
        :param bounds: test_sequence of bounds tuples; e.g., [(1,2),(3,4)]
        :param expected_num_mutations: on average how many mutations done (specificy either this or probability, but not both)
        :param probability: the probability of mutating any given gene (specificy either this or expected_num_mutations, but not both)
    """
    assert(bool(expected_num_mutations is not None) ^ bool(probability is not None)), f"Got expected_num_mutations={expected_num_mutations} and probability={probability}.  One must be specified, but not both."
    assert((probability is None) or (probability >= 0))
    assert((probability is None) or (probability <= 1))

    def randomint_mutate(value, bound, probability):
        """ mutate an integer given a probability
        """
        if random.random() < probability:
            return random.randint(*bound)
        else:
            return value

    if probability is None:
        p = compute_expected_probability(expected_num_mutations, genome)
    else:
        p = probability

    genome = [randomint_mutate(gene, bound, p) for gene, bound in
              zip(genome, bounds)]

    return genome



##############################
# Function mutate_binomial
##############################
@curry
@iteriter_op
def mutate_binomial(next_individual: Iterator, std: float, bounds: list,
                    expected_num_mutations: float = None,
                    probability: float = None,
                    n: int = 10000) -> Iterator:
    """Mutate genes by adding an integer offset sampled from a binomial distribution
    centered on the current gene value.

    This is very similar to applying additive Gaussian mutation and then rounding to
    the nearest integer, but does so in a way that is more natural for integer-valued
    genes.

    :param float std: standard deviation of the binomial distribution
    :param bounds: list of pairs of hard bounds to clip each gene by (to prevent mutation from
        carrying a gene value outside an allowed range)
    :param expected_num_mutations: on average how many mutations done (specificy either this or probability, but not both)
    :param probability: the probability of mutating any given gene (specificy either this or expected_num_mutations, but not both)
    :param int n: the number of "coin flips" to use in the binomial process (defaults to 10000)

    Usage example:

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.int_rep.ops import mutate_binomial
    >>> population = iter([ Individual([1,1]) ])
    >>> operator = mutate_binomial(std=2.5,
    ...                            bounds=[(0, 10), (0, 10)],
    ...                            expected_num_mutations=1)
    >>> mutated = next(operator(population))

    .. note::
        The binomial distribution is defined by two parameters, `n` and `p`.  Here we 
        simplify the interface by asking instead for an `std` parameter, and fixing
        a high value of `n` by default.  The value of `p` needed to obtain the
        given `std` is computed for you internally.

        As the plots below illustrate, the binomial distribution is approximated by a
        Gaussian.  For high `n` and large standard deviations, the two are effectively
        equivalent.  But when the standard deviation (and thus binomial `p` parameter) 
        is relatively small, the approximation becomes less accurate, and the binomial
        differs somewhat from a Gaussian.

        .. plot::

            from matplotlib import pyplot as plt
            import numpy as np
            from scipy.special import comb

            def binomial_p_from_std(n, std):
                if (4*std**2/n > 1):
                    raise ValueError(f"The provided value of n ({n}) is too low to support a Binomial distribution with a standard deviation of {std}.  Choose a higher value of n, or reduce the std.")
                return (1 - np.sqrt(1-4*std**2/n))/2

            def binomial_pmf(k, n, p):
                return comb(n, k)*p**k*(1-p)**(n-k)

            def gaussian_pmf(x, m, s):
                return 1/(np.sqrt(2*np.pi)*s) * np.exp(-1/2*((x - m)/s)**2)

            std=1.1
            x = np.arange(int(-10*std), int(10*std))

            n = 15
            p = binomial_p_from_std(n, std)
            print(f"n={n}, p={p}")
            y = [ binomial_pmf(n*p - k, n, p) for k in x]
            plt.plot(x, y, color='gray', label=f"Binomial(n={n})")

            n = 10000
            p = binomial_p_from_std(n, std)
            print(f"n={n}, p={p}")
            y = [ binomial_pmf(n*p - k, n, p) for k in x]
            plt.plot(x, y, color='red', label=f"Binomial(n={n})")

            y = [ gaussian_pmf(k, 0, std) for k in x]
            plt.plot(x, y, linestyle='dashed', label="Gaussian")

            plt.gca().legend()
    """
    if (expected_num_mutations is not None) and (probability is not None):
        raise ValueError("Received parameters for 'probability' and 'expected_num_mutations', but can only use one or the other.")
    if (expected_num_mutations is None) and (probability is None):
        raise ValueError("Received no value for 'probability' or 'expected_num_mutations'.  Must have one.")
    if (probability is not None) and ((probability < 0) or (probability > 1)):
        raise ValueError(f"The value of 'probability' is {probability}, but must be >= 0 and <= 1.")
    
    while True:
        try:
            individual = next(next_individual)
        except StopIteration:
            return

        individual.genome = individual_mutate_binomial(individual.genome, std, bounds,
                                                      expected_num_mutations=expected_num_mutations,
                                                      probability=probability)

        individual.fitness = None  # invalidate fitness since we have new genome

        yield individual



##############################
# Function individual_mutate_binomial
##############################
def individual_mutate_binomial(genome: list,
                               std: float,
                               bounds: list,
                               expected_num_mutations: float = None,
                               probability: float = None,
                               n: int = 10000,) -> list:
    """
    Perform additive binomial mutation of a particular genome.

    >>> genome = [42, 12]
    >>> bounds = [(0,50), (-10,20)]
    >>> new_genome = individual_mutate_binomial(genome, std=0.5, bounds=bounds,
    ...                                         expected_num_mutations=1)

    """
    assert(bool(expected_num_mutations is not None) ^ bool(probability is not None)), f"Got expected_num_mutations={expected_num_mutations} and probability={probability}.  One must be specified, but not both."
    assert((probability is None) or (probability >= 0))
    assert((probability is None) or (probability <= 1))

    def binomial_mutate(value, p, bound, probability):
        """Mutate an integer by adding a binomial value."""
        assert(len(bound) == 2)
        if random.random() < probability:
            binom_mean = n*p
            mutated = value + np.random.binomial(n, p) - int(binom_mean)
            low, high = bound
            clipped = max(low, min(high, mutated))
            return clipped
        else:
            return value

    if probability is None:
        probability = compute_expected_probability(expected_num_mutations, genome)
    else:
        probability = probability

    p = _binomial_p_from_std(n, std)
    genome = [binomial_mutate(gene, p, bound, probability) for gene, bound in zip(genome,bounds)]

    return genome
    

def _binomial_p_from_std(n, std):
    """Given a number of 'coin flips' n, compute the value of p that is
    needed to achieve a desired standard deviation."""
    if (4*std**2/n > 1):
        raise ValueError(f"The provided value of n ({n}) is too low to support a Binomial distribution with a standard deviation of {std}.  Choose a higher value of n, or reduce the std.")
    # We arrived at this expression by noting that σ^2 = np(1-p)
    # and solving for p via the quadratic formula
    return (1 - np.sqrt(1-4*std**2/n))/2