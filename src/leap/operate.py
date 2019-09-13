import abc

import numpy as np
from toolz import curry

from leap.core import Individual


##############################
# do_pipeline method
##############################
def do_pipeline(population, context, *pipeline):
    """"""
    for op in pipeline:
        population = op(population, context)
    return population


##############################
# Class Operator
##############################
class Operator(abc.ABC):
    """Abstract base class that documents the interface for operators in a LEAP pipeline.

    LEAP treats operators as functions of two arguments: the population, and a "context" `dict` that may be used in
    some algorithms to maintain some global state or parameters independent of the population.

    You can inherit from this class to define operators as classes.  Classes support operators that take extra arguments
    at construction time (such as a mutation rate) and maintain some internal private state, and they allow certain
    special patterns (such as multi-function operators).

    But inheriting from this class is optional.  LEAP can treat any `callable` object that takes two parameters as an
    operator.  You may define your custom operators as closures (which also allow for construction-time arguments and
    internal state), as simple functions (when no additional arguments are necessary), or as curried functions (i.e.
    with the help of `toolz.pipe(...)`.

    """

    @abc.abstractmethod
    def __call__(self, population, context):
        """
        The basic interface for a pipeline operator in LEAP.

        :param population: a list of individuals to be operated upon
        :param context: a `dict` of auxiliary state that some operators read from and/or write to
        """
        pass


##############################
# evaluate operator
##############################
def evaluate(population, context=None):
    """
    Evaluate all the individuals in the given population

    :param population: a list of individuals to be evaluated
    :param context: pipeline context object (ignored)
    :return: the evaluated individuals
    """
    for individual in population:
        individual.evaluate()

    return population


##############################
# cloning operator
##############################
@curry
def cloning(population, context=None, offspring_per_ind=1):
    """
    >>> from leap.util import print_list
    >>> pop = [Individual([1, 2]),
    ...        Individual([3, 4]),
    ...        Individual([5, 6])]
    >>> new_pop = cloning(pop)
    >>> print_list(new_pop)
    [[1, 2], [3, 4], [5, 6]]

    If we edit individuals in the original, new_pop shouldn't change:

    >>> pop[0].genome[1] = 7
    >>> pop[2].genome[0] = 0
    >>> print_list(pop)
    [[1, 7], [3, 4], [0, 6]]

    >>> print_list(new_pop)
    [[1, 2], [3, 4], [5, 6]]

    If we set `offspring_per_ind`, we can create bigger populations:

    >>> pop = [Individual([1, 2]),
    ...        Individual([3, 4]),
    ...        Individual([5, 6])]
    >>> new_pop = cloning(pop, offspring_per_ind=3)
    >>> print_list(new_pop)
    [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [5, 6], [5, 6], [5, 6]]
    """
    assert(population is not None)
    assert(offspring_per_ind > 0)

    result = []
    for ind in population:
        for i in range(offspring_per_ind):
            result.append(ind.clone())

    assert(len(result) == offspring_per_ind*len(population))
    return result


##############################
# mutate_bitflip operator
##############################
@curry
def mutate_bitflip(population, context, prob):
    """
    >>> from leap.util import print_list
    >>> population = [Individual(genome=[1, 0, 1, 1, 0])]
    >>> always = mutate_bitflip(prob=1.0)
    >>> print_list(always(population, None))
    [[0, 1, 0, 0, 1]]

    Individuals are modified in place:

    >>> print_list(population)
    [[0, 1, 0, 0, 1]]

    >>> population = [Individual(genome=[1, 0, 1, 1, 0])]
    >>> never = mutate_bitflip(prob=0.0)
    >>> print_list(never(population, None))
    [[1, 0, 1, 1, 0]]
    """
    def flip(x):
        if np.random.uniform() < prob:
            return 0 if x == 1 else 1
        else:
            return x

    result = []
    for ind in population:
        ind.genome = [flip(x) for x in ind.genome]
        ind.fitness = None
        result.append(ind)
    return result


##############################
# mutate_gaussian operator
##############################
@curry
def mutate_gaussian(population, context, prob, std):
    def add_gauss(x):
        if np.random.uniform() < prob:
            return x + np.random.normal()*std
        else:
            return x

    result = []
    for ind in population:
        ind.genome = [add_gauss(x) for x in ind.genome]
        ind.fitness = None
        result.append(ind)
    return result


##############################
# truncation selection operator
##############################
@curry
def truncation(population, context, mu):
    """
    Returns the `mu` individuals with the best fitness.

    For example, say we have a population of 10 individuals with the following fitnesses:

    >>> fitnesses = [0.12473057, 0.74763715, 0.6497458 , 0.36178902, 0.41318757, 0.69130493, 0.67464942, 0.14895497, 0.15406642, 0.31307095]
    >>> population = [Individual([i]) for i in range(10)]
    >>> for (ind, f) in zip(population, fitnesses):
    ...     ind.fitness = f

    The three highest-fitness individuals are are the indices 1, 5, and 6:

    >>> from leap.util import print_list
    >>> print_list(truncation(population, None, 3))
    [[1], [5], [6]]
    """
    inds = list(sorted(list(population), key=lambda x: x.fitness, reverse=True))
    return inds[0:mu]


##############################
# tournament selection operator
##############################
@curry
def tournament(population, context, n, num_competitors=2):
    """
    Select `n` individuals form a population via tournament selection.
    :param list population: A list of :py:class:`Individual`s
    :param int n: The number of individuals to select
    :param int num_competitors: The number of individuals that compete in each tournament
    :return: A generator that produces `n` individuals
    >>> pop = [Individual(genome=[1, 0, 1, 1, 0]), \
               Individual(genome=[0, 0, 1, 0, 0]), \
               Individual(genome=[0, 1, 1, 1, 1]), \
               Individual(genome=[1, 0, 0, 0, 1])]
    >>> for (ind, f) in zip(pop, [3, 1, 4, 2]):
    ...     ind.fitness = f
    >>> tournament(pop, None, 3) # doctest:+ELLIPSIS
    [..., ..., ...]
    """
    result = []
    for i in range(n):
        competitors = np.random.choice(population, num_competitors)
        result.append(max(competitors, key=lambda x: x.fitness))
    return result


##############################
# best  method
##############################
def best(population, context=None):
    """
    Syntactic sugar to select the best individual in a population.

    :param population: a list of individuals
    :param context: optional `dict` of auxiliary state (ignored)

    >>> from leap import core, binary
    >>> pop = [Individual([1, 0, 1, 1, 0], core.IdentityDecoder(), binary.MaxOnes()), \
               Individual([0, 0, 1, 0, 0], core.IdentityDecoder(), binary.MaxOnes()), \
               Individual([0, 1, 1, 1, 1], core.IdentityDecoder(), binary.MaxOnes()), \
               Individual([1, 0, 0, 0, 1], core.IdentityDecoder(), binary.MaxOnes())]
    >>> pop = evaluate(pop)
    >>> print(best(pop))
    [0, 1, 1, 1, 1]
    """
    assert(len(population) > 0)
    return max(population)
