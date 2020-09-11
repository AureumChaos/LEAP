#!/usr/bin/env python3
"""
    Classes related to individuals that represent posed solutions.


    TODO Need to decide if the logic in __eq__ and __lt__ is overly complex.
    I like that this reduces the dependency on Individuals on a Problem
    (because sometimes you have a super simple situation that doesn't require
    explicitly couching your problem in a Problem subclass.)
"""
from math import nan
import abc
from copy import deepcopy
from functools import total_ordering
import random

from toolz import curry
from toolz.itertoolz import pluck

from leap_ec import util

# This defines a global context that is a dictionary of dictionaries.  The
# intent is for certain operators and functions to add to and modify this
# context.  Third party operators and functions will just add a new top-level
# dedicated key.
# context['leap'] is for storing general LEAP running state, such as current
#    generation.
# context['leap']['distributed'] is for storing leap.distributed running state
# context['leap']['distributed']['non_viable'] accumulates counts of non-viable
#    individuals during distributed.eval_pool() and
#    distributed.async_eval_pool() runs.
context = {'leap': {'distributed': {'non_viable': 0}}}





##############################
# Class Representation
##############################
class Representation():
    """A `Representation` is a simple data structure that wraps the
    components needed to define, initialize, and decode individuals.

    This just serves as some syntactic sugar when we are specifying
    algorithms---so that representation-related components are grouped
    together and clearly labeled `Representation`. """

    def __init__(self, decoder, initialize, individual_cls=Individual):
        self.decoder = decoder
        self.initialize = initialize
        self.individual_cls = individual_cls

    def create_population(self, pop_size, problem):
        """ make a new population

        :param pop_size: how many individuals should be in the population
        :param problem: to be solved
        :return: a population of `individual_cls` individuals
        """
        return self.individual_cls.create_population(pop_size,
                                                     initialize=self.initialize,
                                                     decoder=self.decoder,
                                                     problem=problem)


