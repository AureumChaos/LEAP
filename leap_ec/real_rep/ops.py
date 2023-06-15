#!/usr/bin/env python3
"""
    Pipeline operators for real-valued representations
"""
import math
from collections.abc import Iterable
from typing import Iterator, List, Tuple, Union

import numpy as np

from leap_ec.util import wrap_curry
from leap_ec.ops import compute_expected_probability, iteriter_op, random_bernoulli_vector


##############################
# Function mutate_gaussian
##############################
@wrap_curry
@iteriter_op
def mutate_gaussian(next_individual: Iterator,
                    std,
                    expected_num_mutations: Union[int, str] = None,
                    bounds=(-math.inf, math.inf),
                    transform_slope: float = 1.0,
                    transform_intercept: float = 0.0) -> Iterator:
    """Mutate and return an Individual with a real-valued representation.

    This operators on an iterator of Individuals:

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.real_rep.ops import mutate_gaussian
    >>> import numpy as np
    >>> pop = iter([Individual(np.array([1.0, 0.0]))])

    Mutation can either use the same parameters for all genes:

    >>> op = mutate_gaussian(std=1.0, expected_num_mutations='isotropic', bounds=(-5, 5))
    >>> mutated = next(op(pop))

    Or we can specify the `std` and `bounds` independently for each gene:

    >>> pop = iter([Individual(np.array([1.0, 0.0]))])
    >>> op = mutate_gaussian(std=[0.5, 1.0],
    ...                      expected_num_mutations='isotropic',
    ...                      bounds=[(-1, 1), (-10, 10)]
    ... )
    >>> mutated = next(op(pop))

    :param next_individual: to be mutated
    :param std: standard deviation to be equally applied to all individuals;
        this can be a scalar value or a "shadow vector" of standard deviations
    :param expected_num_mutations: if an int, the *expected* number of mutations per
        individual, on average.  If 'isotropic', all genes will be mutated.
    :param bounds: to clip for mutations; defaults to (- ∞, ∞)
    :return: a generator of mutated individuals.
    """
    if expected_num_mutations is None:
        raise ValueError("No value given for expected_num_mutations.  Must be either a float or the string 'isotropic'.")
    
    genome_mutator = genome_mutate_gaussian(std=std,
                                expected_num_mutations=expected_num_mutations,
                                bounds=bounds,
                                transform_slope=transform_slope,
                                transform_intercept=transform_intercept)

    while True:
        individual = next(next_individual)

        individual.genome = genome_mutator(individual.genome)
        # invalidate fitness since we have new genome
        individual.fitness = None

        yield individual


##############################
# Function genome_mutate_gaussian
##############################
@wrap_curry
def genome_mutate_gaussian(genome,
                           std: float,
                           expected_num_mutations,
                           bounds: Tuple[float, float] =
                             (-math.inf, math.inf),
                           transform_slope: float = 1.0,
                           transform_intercept: float = 0.0):
    """Perform Gaussian mutation directly on real-valued genes (rather than
    on an Individual).

    This used to be inside `mutate_gaussian`, but was moved outside it so that
    `leap_ec.segmented.ops.apply_mutation` could directly use this function,
    thus saving us from doing a copy-n-paste of the same code to the segmented
    sub-package.

    :param genome: of real-valued numbers that will potentially be mutated
    :param std: the mutation width—either a single float that will be used for
        all genes, or a list of floats specifying the mutation width for
        each gene individually.
    :param expected_num_mutations: on average how many mutations are expected
    :return: mutated genome
    """
    assert(std is not None)
    assert(isinstance(std, Iterable) or (std >= 0.0))
    assert(expected_num_mutations is not None)

    if isinstance(std, Iterable):
        std = np.array(std)

    # compute actual probability of mutation based on expected number of
    # mutations and the genome length

    if not isinstance(genome, np.ndarray):
        raise ValueError(("Expected genome to be a numpy array. "
                        f"Got {type(genome)}."))

    if expected_num_mutations == 'isotropic':
        # Default to isotropic Gaussian mutation
        p = 1.0
    else:
        p = compute_expected_probability(expected_num_mutations, genome)

    # select which indices to mutate at random
    indices_to_mutate = random_bernoulli_vector(shape=genome.shape, p=p)

    # Pick out just the std values we need for the mutated genes
    std_selected = std if not isinstance(std, Iterable) else std[indices_to_mutate]

    # Apply additive Gaussian noise to the selected genes
    new_gene_values = transform_slope * (genome[indices_to_mutate] \
                                                    + np.random.normal(size=sum(indices_to_mutate)) \
                                                    # scalar multiply if scalar; element-wise if std is an ndarray
                                                    * std_selected) \
                                + transform_intercept
    genome[indices_to_mutate] = new_gene_values

    # Implement hard bounds
    genome = apply_hard_bounds(genome, bounds)

    return genome


##############################
# Function apply_hard_bounds
##############################
def apply_hard_bounds(genome, hard_bounds):
    """A helper that ensures that every gene is contained within the given bounds.

    :param genome: list of values to apply bounds to.
    :param hard_bounds: if a `(low, high)` tuple, the same bounds will be used for every gene.
        If a list of tuples is given, then the ith bounds will be applied to the ith gene.

    Both sides of the range are inclusive:

    >>> genome = np.array([0, 10, 20, 30, 40, 50])
    >>> apply_hard_bounds(genome, hard_bounds=(20, 40))
    array([20, 20, 20, 30, 40, 40])

    Different bounds can be used for each locus by passing in a list of tuples:

    >>> bounds= [ (0, 1), (0, 1), (50, 100), (50, 100), (0, 100), (0, 10) ]
    >>> apply_hard_bounds(genome, hard_bounds=bounds)
    array([ 0,  1, 50, 50, 40, 10])
    """
    assert(genome is not None)
    assert(isinstance(genome, Iterable))
    assert(hard_bounds is not None)

    if not isinstance(hard_bounds[0], Iterable):
        # scalar bounds apply to each gene
        low = hard_bounds[0]
        high = hard_bounds[1]
    elif isinstance(hard_bounds, np.ndarray):
        low = hard_bounds[:, 0]
        high = hard_bounds[:, 1]
    else:
        # Looping through twice here is faster than converting to
        # numpy array and slicing for the column
        low = [bound[0] for bound in hard_bounds]
        high = [bound[1] for bound in hard_bounds]

    return np.clip(genome, a_min=low, a_max=high)
