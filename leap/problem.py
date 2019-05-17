#!/usr/bin/env python3
"""
    A Problem encapsulates a particular problem that we wish to solve. Individuals are bound to a problem, which
    knows how to evaluate and compare individuals.  A given problem may decode an individual's Encoding to
    evaluate that individual's fitness.
"""

from abc import ABCMeta, abstractmethod
from smallLEAP.reproduction import create_binary_sequence

from smallLEAP.individual import Individual


def evaluate(population):
    """
    Evaluate all the individuals in the given population

    TODO could be moved to individual.py
    TODO could be re-written in functional programming style

    :param population: to be evaluated
    :return: the evaluated individuals
    """
    for individual in population:
        individual.evaluate()

    return population


def evaluate_generator(next_individual):
    """
    TODO could be moved to individual.py

    :param next_individual: is the next individual to be evaluated
    :return:
    """
    while True:
        individual = next(next_individual)
        individual.evaluate()
        yield individual


class Problem(metaclass=ABCMeta):
    """
        Abstract Base Class used to define problem definitions

        TODO Do we want to have a "maximization" boolean that's true if this
        is a maximization problem?  And the less_than and equal functions
        change behavior accordingly?
    """

    def __init__(self):
        super().__init__()


    @abstractmethod
    def evaluate(self, individual):
        """
        Decode and evaluate the given individual based on its genome.

        Practicitioners *must* over-ride this member function.

        Note that by default the individual comparison operators assume a
        maximization problem; if this is a minimization problem, then just
        negate the value when returning the fitness.

        :param individual:
        :return: fitness
        """
        raise NotImplementedError


    def worse_than(self, first, second):
        """
            Used in Individual.__lt__().

            By default returnss first.fitness < second.fitness.  Please
            over-ride if this does not hold for your problem.

            :return: true if the first individual is less fit than the second
        """
        # TODO add check that first and second are of type Individual
        return first.fitness < second.fitness


    def same_as(self, first, second):
        """
            Used in Individual.__eq__().

            By default returnss first.fitness== second.fitness.  Please
            over-ride if this does not hold for your problem.

            :return: true if the first individual is equal to the second
        """
        # TODO add check that first and second are of type Individual
        return first.fitness == second.fitness



def max_ones(individual):
    """ for individual the fitness is the number of genes of value '1'

    Presumes that individual is a binary vector.

    TODO This should be eventually moved to within MaxOnes.  Maybe?  Not
    moving it now because it would be too disruptive of Jeff's work with the
    crossover operator.

    :param individual: to be evaluated
    :return: number of ones in genome
    """
    return individual.count(1)



class MaxOnes(Problem):
    """
    Implementation of MAX ONES problem where the individuals are represented
    by a bit vector

    We don't need an encoder since the raw genome is *already* in the phenotypic space.

    """
    def __init__(self):
        """
        Create a MAX ONES problem with individuals that have bit vectors of
        size `length`
        """
        super().__init__()


    def evaluate(self, individual):
        binary_sequence = individual.decode()
        return max_ones(binary_sequence)



class Spheroid(Problem):
    """ Classic spheroid problem
    """
    def __init__(self):
        super().__init__()

    def worse_than(self, first, second):
        """
            Used in Individual.__lt__().

            Since this is a maximization problem, return first.fitness > second.fitness

            :return: true if the first individual is less fit than the second
        """
        # TODO add check that first and second are of type Individual
        return first.fitness > second.fitness

    def evaluate(self, individual):
        """

        :param individual: to be evaluated
        :return: sum(individual.genome**2)
        """
        return sum([x**2 for x in individual.decode()])
