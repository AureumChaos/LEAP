#!/usr/bin/env python3
"""
    Segmented representation specific pipeline operators.
"""
from typing import Iterator, Callable
import random
from toolz import curry

from leap_ec.ops import compute_expected_probability, iteriter_op


##############################
# Function apply_mutation
##############################
@curry
@iteriter_op
def apply_mutation(next_individual: Iterator,
                   mutator: Callable[[list, float], list],
                   expected_num_mutations: float = 1.0) -> Iterator:
    """
    This expects next_individual to have a segmented representation; i.e.,
    a test_sequence of sequences.  `mutator_func` will be applied to each
    sub-test_sequence with the expected probability.  The expected probability
    applies to *all* the sequences, and defaults to a single mutation among
    all components, on average.

    >>> from leap_ec.binary_rep.ops import genome_mutate_bitflip
    >>> mutation_op = apply_mutation(mutator=genome_mutate_bitflip)

    >>> from leap_ec.individual import Individual
    >>> original = Individual([[0,0],[1,1]])
    >>> mutated = next(mutation_op(iter([original])))

    :param next_individual: to possibly mutate
    :param mutator: function to be applied to each segment in the
        individual's genome; first argument is a segment, the second the
        expected probability of mutating each segment element.
    :param expected: expected mutations on average in [0.0,1.0]
    :return: yielded mutated individual
    """
    while True:
        individual = next(next_individual)

        # compute expected probability of mutation _per segment_
        per_segment_expected_prob = compute_expected_probability(
            expected_num_mutations, individual.genome)

        # Apply mutation function using the expected probability to create a
        # new test_sequence of sequences to be assigned to the genome.
        mutated_genome = [mutator(segment,
                                  expected_num_mutations=per_segment_expected_prob)
                          for segment in individual.genome]

        individual.genome = mutated_genome

        # invalidate the fitness since we have a modified genome
        individual.fitness = None

        yield individual


##############################
# add_segment
##############################
@curry
@iteriter_op
def add_segment(next_individual: Iterator,
                seq_initializer: Callable,
                probability: float,
                append: bool = False) -> Iterator:
    """ Possibly add a segment to the given individual

    New segments can be always appended, or randomly inserted within the
    individual's genome.

    TODO add a parameter for accepting a function that will yield a distribution
    for the number of segments to be randomly inserted.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.binary_rep.initializers import create_binary_sequence
    >>> original = Individual([[0,0],[1,1]])
    >>> mutated = next(add_segment(iter([original]), seq_initializer=create_binary_sequence(2), probability=1.0))

    :param next_individual: to possibly add a segment
    :param seq_initializer: callable for initializing any new segments
    :param probability: likelihood of adding a segment
    :param append: if True, always append any new segments
    :return: yielded individual with a possible new segment
    """
    while True:
        individual = next(next_individual)

        if random.random() < probability:
            new_segment = seq_initializer()

            if append:
                individual.genome.append(new_segment)
            else:
                # + 1 to allow for appending new segment
                insertion_point = random.randrange(len(individual.genome) + 1)
                individual.genome.insert(insertion_point, new_segment)

            # invalidate the fitness since we have a modified genome
            individual.fitness = None

        yield individual


##############################
# remove_segment
##############################
@curry
@iteriter_op
def remove_segment(next_individual: Iterator,
                   probability: float) -> Iterator:
    """ for some chance, remove a segment

        Nothing happens if the individual has a single segment; i.e., there is
        no chance for an empty individual to be returned.

    >>> from leap_ec.individual import Individual
    >>> original = Individual([[0,0],[1,1]])
    >>> mutated = next(remove_segment(iter([original]), probability=1.0))
    >>> assert mutated.genome == [[0,0]] or mutated.genome == [[1,1]]

        :param next_individual: to have a segment possibly removed
        :param probability: likelihood of removing a segment
        :returns: the next individual
    """
    while True:
        individual = next(next_individual)

        if len(individual.genome) > 1:
            # we ignore empty genomes, or genomes with a single segment

            if random.random() < probability:
                removed_segment = random.randrange(len(individual.genome))
                del individual.genome[removed_segment]

                # invalidate the fitness since we have a modified genome
                individual.fitness = None

        yield individual


##############################
# copy_segment
##############################
@curry
@iteriter_op
def copy_segment(next_individual: Iterator,
                 probability: float,
                 append: bool = False) -> Iterator:
    """ with a given probability, randomly select and copy a segment

    >>> from leap_ec.individual import Individual
    >>> original = Individual([[0,0]])
    >>> mutated = next(copy_segment(iter([original]), probability=1.0))
    >>> assert mutated.genome == [[0,0],[0,0]]

        :param next_individual: to have a segment possibly removed
        :param probability: likelihood of doing this
        :param append: if True, always append any new segments
        :returns: the next individual
    """
    while True:
        individual = next(next_individual)

        if random.random() < probability:
            copied_segment = \
                individual.genome[random.randrange(len(individual.genome))]

            if append:
                individual.genome.insert(len(individual.genome), copied_segment)
            else:
                # + 1 to allow for appending new segment
                insertion_point = random.randrange(len(individual.genome) + 1)
                individual.genome.insert(insertion_point, copied_segment)

            # invalidate the fitness since we have a modified genome
            individual.fitness = None

        yield individual
