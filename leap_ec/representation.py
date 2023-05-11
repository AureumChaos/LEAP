#!/usr/bin/env python3
"""
A `Representation` is a simple data structure that wraps the
components needed to define, initialize, and decode individuals.

This just serves as some syntactic sugar when we are specifying
algorithms---so that representation-related components are grouped
together and clearly labeled `Representation`.
"""
from leap_ec.individual import Individual
from leap_ec.decoder import IdentityDecoder

##############################
# Class Representation
##############################
class Representation():
    """ Syntactic sugar for some of the monolithic functions that
        conveniently combines a decoder, initializer, and an Individual
        class since those always work in tandem, but can still be loosely
        coupled.
     """

    def __init__(self, initialize, decoder=IdentityDecoder(), individual_cls=Individual):
        self.decoder = decoder
        self.initialize = initialize
        self.individual_cls = individual_cls

    def create_individual(self, problem):
        """Make a single individual."""
        return self.create_population(1, problem)[0]

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
