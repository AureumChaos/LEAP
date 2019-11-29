""" Module for fundamental evolutionary operators.  You'll find many
traditional selection and reproduction strategies here.

"""

import abc
import itertools
import random


import numpy as np
import toolz
from toolz import curry, topk

from leap.core import Individual


##############################
# do_pipeline method
##############################
# def do_pipeline(population, context, *pipeline):
#     """
#     FIXME commented out because this isn't used anywhere; will uncomment and
#     modify should that change.  MAC.  10/18/19
#     :param population:
#     :param context:
#     :param pipeline:
#     :return:
#     """
#     for op in pipeline:
#         population, context = op(population, context)
#     return population, context


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
    def __call__(self, population, *args, **kwargs):
        """
        The basic interface for a pipeline operator in LEAP.

        :param *args: optional variable sequence of arguments
        :param **kwargs: optional dictionary of arguments
        :param population: a list of individuals to be operated upon
        """
        pass





##############################
# evaluate operator
##############################
@curry
def evaluate(next_individual, *args, **kwargs):
    """ Evaluate and returns the next individual in the pipeline

    >>> import core, binary_problems

    We need to specify the decoder and problem so that evaluation is possible.

    >>> ind = core.Individual([1,1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())

    >>> evaluated_ind = evaluate(iter([ind]))

    :param next_individual: iterator pointing to next individual to be evaluated
    :return: the evaluated individual
    """
    while True:
        individual, pipe_args, pipe_kwargs = next(next_individual)
        individual.evaluate()

        # Use unpacking to combine args passed in explicitly from the user with
        # those passed through the pipe.
        yield individual, (*pipe_args, *args), {**pipe_kwargs, **kwargs}


##############################
# clone operator
##############################
@curry
def clone(next_individual, *args, **kwargs):
    """ clones and returns the next individual in the pipeline

    >>> import core

    Create a common decoder and problem for individuals.

    >>> original = Individual([1,1])

    >>> cloned_generator = clone(iter([original]))

    :param next_individual: iterator for next individual to be cloned
    :return: copy of next_individual
    """
    while True:
        individual, pipe_args, pipe_kwargs = next(next_individual)
        yield individual.clone(), (*pipe_args, *args), {**pipe_kwargs, **kwargs}


# ##############################
# # mutate_bitflip operator
# ##############################
@curry
def mutate_bitflip(next_individual, expected=1, *args, **kwargs):
    """ mutate and return an individual with a binary representation

    >>> import core, binary_problems

    >>> original = Individual([1,1])

    >>> mutated_generator = mutate_bitflip(iter([original]))

    :param individual: to be mutated
    :param expected: the *expected* number of mutations, on average
    :param args: optional args
    :param kwargs: optional keyword args
    :return: mutated individual
    """
    def flip(gene):
        if random.random() < probability:
            return (gene + 1) % 2
        else:
            return gene

    individual, pipe_args, pipe_kwargs = next(next_individual)

    # Given the average expected number of mutations, calculate the probability
    # for flipping each bit.
    probability = 1.0 / len(individual.genome) * expected

    while True:
        individual.genome = [flip(gene) for gene in individual.genome]

        yield individual,  (*pipe_args, *args), {**pipe_kwargs, **kwargs}


# ##############################
# # mutate_gaussian operator
# ##############################
# @curry
# def mutate_gaussian(population, context, prob, std, hard_bounds=(-np.inf, np.inf)):
#     def add_gauss(x):
#         if np.random.uniform() < prob:
#             return x + np.random.normal()*std
#         else:
#             return x
#
#     def clip(x):
#         return max(hard_bounds[0], min(hard_bounds[1], x))
#
#     result = []
#     for ind in population:
#         ind.genome = [clip(add_gauss(x)) for x in ind.genome]
#         ind.fitness = None
#         result.append(ind)
#     return result, context
#
#
# ##############################
# # truncation selection operator
# ##############################
# @curry
# def truncation(population, context, mu):
#     """
#     Returns the `mu` individuals with the best fitness.
#
#     For example, say we have a population of 10 individuals with the following fitnesses:
#
#     >>> from leap import core, real
#     >>> fitnesses = [0.12473057, 0.74763715, 0.6497458 , 0.36178902, 0.41318757, 0.69130493, 0.67464942, 0.14895497, 0.15406642, 0.31307095]
#     >>> population = [Individual([i], core.IdentityDecoder(), real.Spheroid()) for i in range(10)]
#     >>> for (ind, f) in zip(population, fitnesses):
#     ...     ind.fitness = f
#
#     The three highest-fitness individuals are are the indices 1, 5, and 6:
#
#     >>> from leap.util import print_list
#     >>> pop, _ = truncation(population, None, 3)
#     >>> print_list(pop)
#     [[1], [5], [6]]
#     """
#     inds = list(sorted(list(population), reverse=True))
#     return inds[0:mu], context

def truncate(population, size, second_population=None, *args, **kwargs):
    """ return the `size` best individuals from the given population

        This defaults to (mu, lambda) if `second_population` is not given.

        second_population is an optional population that is "blended" with the
        first for truncation purposes, and is usually used to allow parents
        and offspring to compete. (I.e., mu + lambda).

        FIXME this will also "truncate" *args and **kwargs passed down the line since this expects a sequence
        and not a generator. (Actually, this will be after a pool() call, which returns those, so this should
        work as expected, right?)

        >>> from leap import core, ops, binary_problems
        >>> pop = []
        >>> pop.append(core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
        >>> pop.append(core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
        >>> pop.append(core.Individual([1, 1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
        >>> pop.append(core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

        We need to evaluate them to get their fitness to sort them for truncation.

        >>> pop = [evaluate(i) for i in pop]

        >>> truncated = truncate(pop, 3)

        >>> truncated

        :param population: that needs downsized
        :param size: is what to resize population to
        :param second_population: is optional second population to include
                                  with population for downsizing
        :return: truncated population (plus optional second population)
    """
    if second_population is not None:
        return toolz.itertoolz.topk(size, itertools.chain(population,
                                                          second_population))
    else:
        return toolz.itertoolz.topk(size, population)

# ##############################
# # tournament selection operator
# ##############################
# @curry
# def tournament(population, context, n, num_competitors=2):
#     """
#     Select `n` individuals form a population via tournament selection.
#     :param list population: A list of individuals
#     :param int n: The number of individuals to select
#     :param int num_competitors: The number of individuals that compete in each tournament
#     :return: A generator that produces `n` individuals
#
#     >>> from leap import core, real, data
#     >>> pop = data.test_population
#     >>> for (ind, f) in zip(pop, [3, 1, 4, 2]):
#     ...     ind.fitness = f
#     >>> pop, _ = tournament(pop, None, 3)
#     >>> pop  # doctest:+ELLIPSIS
#     [..., ..., ...]
#     """
#     result = []
#     for i in range(n):
#         competitors = np.random.choice(population, num_competitors)
#         result.append(max(competitors))
#     return result, context
#
#
# ##############################
# # Class MuPlusLambda
# ##############################
# class MuPlusLambdaConcatenation(Operator):
#     def __init__(self):
#         self.parents = None
#
#     def capture_parents(self, population, *args, **kwargs):
#         self.parents = population
#         return population, args, kwargs
#
#     def __call__(self, population, *args, **kwargs):
#         return self.parents + population, args, kwargs


def naive_cyclic_selection_generator(population, *args, **kwargs):
    """ Deterministically returns individuals, and repeats the same sequence
    when exhausted.

    This is "naive" because it doesn't shuffle the population between complete
    tours to minimize bias.

    TODO implement non-naive version that shuffles population before first
    iteration and after every complete loop to minimize sample bias.

    >>> import core, ops

    >>> pop = []

    >>> pop.append(core.Individual([0, 0]))
    >>> pop.append(core.Individual([0, 1]))

    >>> cyclic_selector = ops.naive_cyclic_selection_generator(pop)

    :param population: from which to select
    :return: the next selected individual
    """
    iter = itertools.cycle(population)

    while True:
        yield next(iter), args, kwargs


@curry
def pool(next_individual, size, *args, **kwargs):
    """ 'Sink' for creating `size` individuals from preceding pipeline source.

    Allows for "pooling" individuals to be processed by next pipeline
    operator.  Typically used to collect offspring from preceding set of
    selection and birth operators, but could also be used to, say, "pool"
    individuals to be passed to an EDA as a training set.

    >>> import core, ops

    >>> pop = []

    >>> pop.append(core.Individual([0, 0]))
    >>> pop.append(core.Individual([0, 1]))

    >>> cyclic_selector = ops.naive_cyclic_selection_generator(pop)

    >>> pool = ops.pool(cyclic_selector, 3)

    print(pool)
    [Individual([0, 0], None, None), Individual([0, 1], None, None), Individual([0, 0], None, None)]

    :param next_individual: generator for getting the next offspring
    :param size: how many kids we want
    :return: population of `size` offspring
    """
    # TODO this could be more elegant, and I'm not sure about the priority
    # order for what overwrites what for function arguments vs. pipe data.
    final_args = ()
    final_kwargs = {}
    final_pool = []

    for _ in range(size):
        individual, pipe_args, pipe_kwargs = next(next_individual)
        final_args = (*final_args, *pipe_args)
        final_kwargs = {**final_kwargs, **pipe_kwargs}

        final_pool.append(individual)

    # return [next(next_individual) for _ in range(size)], args, kwargs
    return final_pool, final_args, final_kwargs
