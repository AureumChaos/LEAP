#!/usr/bin/env python3
"""
    Initializers for integer-valued genomes.
"""
import random
from leap_ec.individual import Individual


##############################
# Closure create_real_vector
##############################
def create_int_vector(bounds):
    """
    A closure for initializing lists of integers for int-vector genomes,
    sampled from a uniform distribution.

    Having a closure allows us to just call the returned function N times
    in `Individual.create_population()`.

    TODO Allow either a single tuple or a sequence of tuples for bounds. —Siggy

    :param bounds: a list of (min, max) values bounding the uniform sampline
        of each element

    :return: A function that, when called, generates a random genome.

    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.real_rep.problems import SpheroidProblem
    >>> bounds = [(0, 1), (-5, 5), (-1, 100)]
    >>> population = Individual.create_population(10, create_int_vector(bounds),
    ...                                           decoder=IdentityDecoder(),
    ...                                           problem=SpheroidProblem())
    """
    def create():
        return [random.randint(min_, max_) for min_, max_ in bounds]

    return create
