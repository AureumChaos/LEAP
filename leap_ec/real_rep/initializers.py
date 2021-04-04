#!/usr/bin/env python3
"""
    Initializers for real values.
"""
import numpy as np

from leap_ec.individual import Individual

##############################
# Closure create_real_vector
##############################
def create_real_vector(bounds):
    """
    A closure for initializing lists of real numbers for real-valued genomes,
    sampled from a uniform distribution.

    Having a closure allows us to just call the returned function N times
    in `Individual.create_population()`.

    TODO Allow either a single tuple or a test_sequence of tuples for bounds. —Siggy

    :param bounds: a list of (min, max) values bounding the uniform sampline
        of each element

    :return: A function that, when called, generates a random genome.


    E.g., can be used for `Individual.create_population()`

    >>> from leap_ec.decoder import IdentityDecoder
    >>> from . problems import SpheroidProblem
    >>> bounds = [(0, 1), (0, 1), (-1, 100)]
    >>> population = Individual.create_population(10, create_real_vector(bounds),
    ...                                           decoder=IdentityDecoder(),
    ...                                           problem=SpheroidProblem())

    """

    low = [ l for l, _ in bounds ]
    high = [ h for _, h in bounds ]

    def create():
        return np.random.uniform(low, high, size=len(bounds))

    return create
