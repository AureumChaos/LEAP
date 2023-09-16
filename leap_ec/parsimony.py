#!/usr/bin/env python3
"""
    Parsimony pressure functions.

    These are intended to be used as `key` parameters for selection operators.

    Provided are Koza-style parsimony pressure and lexicographic parsimony
    key functions.
"""
from leap_ec.util import wrap_curry


def lexical_parsimony(ind):
    """ If two fitnesses are the same, break the tie with the smallest genome

    This implements Lexicographical Parsimony Pressure :cite:p:`Luke2002`, which is essentially
    where if the fitnesses of two individuals are close, then break the tie
    with the smallest genome.

    >>> import toolz
    >>> from leap_ec.individual import Individual
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> import leap_ec.ops as ops
    >>> import numpy as np
    >>> problem = MaxOnes()
    >>> pop = [Individual(np.array([0, 0, 0, 1, 1, 1]), problem=problem),
    ...        Individual(np.array([0, 0]), problem=problem),
    ...        Individual(np.array([1, 1]), problem=problem),
    ...        Individual(np.array([1, 1, 1]), problem=problem)]
    >>> pop = Individual.evaluate_population(pop)
    >>> best, = ops.truncation_selection(pop, size=1)
    >>> print(best.genome, best.fitness)
    [0 0 0 1 1 1] 3

    >>> best, = ops.truncation_selection(pop, size=1, key=lexical_parsimony)
    >>> print(best.genome, best.fitness)
    [1 1 1] 3

    :param ind: to be compared
    :return: altered comparison criteria
    """
    if ind.problem.maximize:
        return (ind.fitness, -len(ind.genome))
    else:
        # Because we are using a key function we are bypassing the
        # ScalarProblem.worse_than() that would invert the fitnesses, so we
        # have to do this here by flipping the sign.
        return (-ind.fitness, -len(ind.genome))


@wrap_curry
def koza_parsimony(ind, *, penalty):
    """ Penalize fitness by genome length times a constant, in the style of :cite:t:`Koza1992`.

    >>> import toolz
    >>> from leap_ec.individual import Individual
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> import leap_ec.ops as ops
    >>> import numpy as np
    >>> problem = MaxOnes()
    >>> pop = [Individual(np.array([0, 0, 0, 1, 1, 1]), problem=problem),
    ...        Individual(np.array([0, 0]), problem=problem),
    ...        Individual(np.array([1, 1]), problem=problem),
    ...        Individual(np.array([1, 1, 1]), problem=problem)]
    >>> pop = Individual.evaluate_population(pop)
    >>> best, = ops.truncation_selection(pop, size=1)
    >>> print(best.genome, best.fitness)
    [0 0 0 1 1 1] 3

    >>> best, = ops.truncation_selection(pop, size=1, key=koza_parsimony(penalty=.5))
    >>> print(best.genome, best.fitness)
    [1 1 1] 3

    .. math::
        f_p(x) = f(x) - cl(x)

    Where f(x) is original fitness, c is a penalty constant, and l(x)
    is the genome length.

    :param ind: to be compared
    :param penalty: for denoting penalty strength
    :return: altered comparison criteria
    """
    if ind.problem.maximize:
        biased_fitness = ind.fitness - penalty * len(ind.genome)
    else:
        # Because we are using a key function we are bypassing the
        # ScalarProblem.worse_than() that would invert the fitnesses, so we
        # have to do this here by flipping the sign.
        biased_fitness = - ind.fitness - penalty * len(ind.genome)

    return biased_fitness
