#!/usr/bin/env python3
"""
    Parsimony pressure functions.

    These are intended to be used as `key` parameters for selection operators.

    Provided are Koza-style parsimony pressure and lexicographic parsimony
    key functions.
"""
from toolz import curry

def lexical_parsimony(ind):
    """ If two fitnesses are the same, break the tie with the smallest genome

    This implements Lexicographical Parsimony Pressure, which is essentially
    where if the fitnesses of two individuals are close, then break the tie
    with the smallest genome.

    .. [Luke2002]
        Luke, S., & Panait, L. (2002, July). Lexicographic parsimony pressure.
        In Proceedings of the 4th Annual Conference on Genetic and Evolutionary
        Computation (pp. 829-836).

    :param ind: to be compared
    :return: altered comparison criteria
    """
    # TODO I think this assumes for maximization only, but not if the
    # first argument is handled by the associated Problem, which would
    # know if this was a maximization or minimization problem.  If not
    # we can switch on ind.problem.maximization, but that is only supported
    # by ScalarProblems, which may not be a big deal.
    return (ind.fitness, -len(ind.genome))


@curry
def koza_parsimony(ind, *, constant):
    """ Penalize fitness by genome length times a constant

    .. [Koza1992]
        J. R. Koza. Genetic Programming: On the Programming of
        Computers by Means of Natural Selection. MIT Press, Cambridge, MA, USA,
        1992.

    .. math::
        f_p(x) = f(x) - cl(x)

        Where f(x) is original fitness, c is a penalty constant, and l(x)
        is the genome length.

    :param ind: to be compared
    :param constant: for denoting penalty strength
    :return: altered comparison criteria
    """
    return ind.fitness - constant * len(ind.genome)
