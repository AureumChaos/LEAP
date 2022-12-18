#!/usr/bin/env python3
"""
    Used to initialize binary sequences
"""
import numpy as np

from leap_ec.individual import Individual
from leap_ec.ops import random_bernoulli_vector


##############################
# Closure create_binary_sequence
##############################
def create_binary_sequence(length):
    """
    A closure for initializing a binary sequences for binary genomes.

    :param length: how many genes?

    :return: a function that, when called, generates a binary vector of given
        length

    E.g., can be used for `Individual.create_population`

    >>> from leap_ec.decoder import IdentityDecoder
    >>> from . problems import MaxOnes
    >>> population = Individual.create_population(10, create_binary_sequence(length=10),
    ...                                           decoder=IdentityDecoder(),
    ...                                           problem=MaxOnes())

    """

    def create():
        return random_bernoulli_vector(length)

    return create
