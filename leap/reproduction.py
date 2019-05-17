#!/usr/bin/env python3
"""
    Various reproduction operators and utility functions.
"""
import random
from toolz import curry

from functools import wraps
from copy import deepcopy

from . import individual


def create_binary_sequence(length=5):
    """ for creating a binary sequences for binary genomes

    :param length: how many genes?
    :return: binary vector of given length
    """
    return [random.choice([0,1]) for _ in range(length)]


def create_real_value_sequence(length, lower_bound, upper_bound):
    """ for creating very basic real-value sequences for real-value genomes

    :param length: how many genes?
    :param lower_bound:
    :param upper_bound:
    :return: real-value vector of requested length
    """
    return [random.uniform(lower_bound, upper_bound) for _ in range(length)]


def binary_flip_mutation(individual, probability):
    """
        Presumes a binary representation in individual.sequence.

        :param individual.encoding.sequence: is collection of bits
        :param probability: is likelihood of flipping an individual bit

        :return: a copy of individual with with individual.sequence bits flipped
                 based on probability
    """
    def flip(gene):
        if random.random() < probability:
            return (gene + 1) % 2
        else:
            return gene

    individual.encoding.sequence = [flip(gene) for gene in individual.encoding.sequence]

    return individual


def bit_flip_mutation_generator(next_individual, probability=0.1):
    """ Generator for mutating an individual and passing it down the line

    :param next_individual: where we get the next individual
    :param probability: how likely are we to mutate each gene
    :return: a mutated individual
    """
    while True:
        yield binary_flip_mutation(next(next_individual), probability)


def create_real_uniform_sequence(length=5, min=0.0, max=1.0):
    """ for creating real value sequences for real-value geneomes


        :param length: how many genes?
        :param min: the lower bound
        :param max: the exclusive upper bound
        :return: real value vector of given length with elements in [min,max)
    """
    return [random.uniform(min, max) for _ in range(length)]


@curry
def static_gaussian_mutation(individual, std=1.0):
    """
    :param individual: to be zapped; presume real-value encoding
    :param std: standard deviation of course
    :return: mutated individual
    """
    individual.encoding.sequence = [random.gauss(gene, std) for gene in individual.encoding.sequence]

    return individual

@curry
def static_gaussian_mutation_generator(next_individual, std=1.0):
    while True:
        yield static_gaussian_mutation(next(next_individual), std)


def uniform_recombination(ind1, ind2, p_swap=0.5):
    """
    Recombination operator that can potentially swap any matching pair of
    genes between two individuals with some probability.

    It is assumed that ind1.genome and ind2.genome are lists of things.

    :param ind1: The first individual
    :param ind2: The second individual
    :param p_swap:

    TODO just return ind1, ind2 if they are identical as can happen with selection with replacement.

    :return: a copy of both individuals with individual.genome bits
             swapped based on probability
    """

    # Check for errors.
    # XXX Are asserts Kosher for error detection?
    assert(len(ind1.encoding.sequence) == len(ind2.encoding.sequence))

    # Perform the crossover
    # There may be more pythonic ways of doing this, but I think they're
    # overly complicated.
    for i in range(len(ind1.encoding.sequence)):
        if random.random() < p_swap:
            ind1.encoding.sequence[i], ind2.encoding.sequence[i] = ind2.encoding.sequence[i], ind1.encoding.sequence[i]

    return (ind1, ind2)


def uniform_recombination_generator(next_individual, p_swap=0.5):
    """
    Generator for recombining two individuals and passing them down the line.

    :param next_individual: where we get the next individual
    :param p_swap: how likely are we to swap each pair of genes
    :return: two recombined individuals
    """
    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)

        child1, child2 = uniform_recombination(parent1, parent2, p_swap)

        yield child1
        yield child2


def create_pool(next_individual, size):
    """ 'Sink' for creating `size` individuals from preceding pipeline source.

    Allows for "pooling" individuals to be processed by next pipeline
    operator.  Typically used to collect offspring from preceding set of
    selection and birth operators, but could also be used to, say, "pool"
    individuals to be passed to an EDA as a training set.

    :param next_individual: generator for getting the next offspring
    :param size: how many kids we want
    :return: population of `size` offspring
    """
    return [next(next_individual) for _ in range(size)]


@curry
def clone_generator(next_individual):
    """ generator version of individual.clone()

    :param next_individual: iterator for next individual to be cloned
    :return: copy of next_individual
    """
    while True:
        yield next(next_individual).clone()
