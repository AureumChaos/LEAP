import random
from typing import Iterator

from toolz import curry

from leap_ec.ops import compute_expected_probability, iteriter_op


##############################
# Function mutate_randint
##############################
@curry
@iteriter_op
def mutate_randint(next_individual: Iterator, bounds,
                   expected_num_mutations: float = 1) -> Iterator:
    """Perform randint mutation on each individual in an iterator (population).

    This operator replaces randomly selected genes with an integer samples
    from a uniform distribution.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.int_rep.ops import mutate_randint

    >>> population = iter([ Individual([1,1]) ])
    >>> operator = mutate_randint(bounds=[(0, 10), (0, 10)])
    >>> mutated = next(operator(population))
    """
    while True:
        try:
            individual = next(next_individual)
        except StopIteration:
            return

        individual.genome = individual_mutate_randint(individual.genome, bounds,
                                                   expected_num_mutations=expected_num_mutations)

        individual.fitness = None  # invalidate fitness since we have new genome

        yield individual


##############################
# Function individual_mutate_randint
##############################
@curry
def individual_mutate_randint(genome: list,
                              bounds: list,
                              expected_num_mutations: float = 1) -> list:
    """ Perform random-integer mutation on a particular genome.

        >>> genome = [42, 12]
        >>> bounds = [(0,50), (-10,20)]
        >>> new_genome = individual_mutate_randint(genome, bounds)

        :param genome: test_sequence of integers to be mutated
        :param bounds: test_sequence of bounds tuples; e.g., [(1,2),(3,4)]
        :param expected_num_mutations: on average how many mutations done
    """
    def randomint_mutate(value, bound, probability):
        """ mutate an integer given a probability
        """
        if random.random() < probability:
            return random.randint(*bound)
        else:
            return value

    probability = compute_expected_probability(expected_num_mutations, genome)

    genome = [randomint_mutate(gene, bound, probability) for gene, bound in zip(genome,bounds)]

    return genome
