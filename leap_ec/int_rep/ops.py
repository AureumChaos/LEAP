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
def individual_mutate_randint(genome: Iterator, bounds,
                              expected_num_mutations: float = 1) -> Iterator:
    """Perform random-integer mutation on a particular genome."""
    probability = compute_expected_probability(expected_num_mutations,
                                               genome)

    def randomint_mutate(gene_index, value, probability):
        """randomint-mutate a bit given a probability
        """
        if random.random() < probability:
            return random.randint(*bounds[gene_index])
        else:
            return value

    return [randomint_mutate(i, gene, probability) for i, gene in enumerate(genome)]
