""" Module for fundamental evolutionary operators.  You'll find many
traditional selection and reproduction strategies here.

"""

import abc
import itertools
import random
import types
from collections.abc import Iterator

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

    TODO The above description is outdated. --Siggy

    You can inherit from this class to define operators as classes.  Classes support operators that take extra arguments
    at construction time (such as a mutation rate) and maintain some internal private state, and they allow certain
    special patterns (such as multi-function operators).

    But inheriting from this class is optional.  LEAP can treat any `callable` object that takes two parameters as an
    operator.  You may define your custom operators as closures (which also allow for construction-time arguments and
    internal state), as simple functions (when no additional arguments are necessary), or as curried functions (i.e.
    with the help of `toolz.curry(...)`.

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
def evaluate(next_individual):
    """ Evaluate and returns the next individual in the pipeline

    >>> from leap import core, binary_problems

    We need to specify the decoder and problem so that evaluation is possible.

    >>> ind = core.Individual([1,1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())

    >>> evaluated_ind = next(evaluate(iter([ind])))

    :param next_individual: iterator pointing to next individual to be evaluated
    :param kwargs: contains optional context state to pass down the pipeline in context dictionaries
    :return: the evaluated individual
    """
    while True:
        # "combined" means combining any args, kwargs passed in to this function with those passed in from upstream
        # in the pipeline.
        # individual, pipe_args, pipe_kwargs = next(next_individual)
        individual = next(next_individual)
        individual.evaluate()

        yield individual


##############################
# clone operator
##############################
@curry
def clone(next_individual):
    """ clones and returns the next individual in the pipeline

    >>> from leap import core

    Create a common decoder and problem for individuals.

    >>> original = Individual([1,1])

    >>> cloned_generator = clone(iter([original]))

    :param next_individual: iterator for next individual to be cloned
    :return: copy of next_individual
    """
    while True:
        individual = next(next_individual)

        yield individual.clone()


# ##############################
# # mutate_bitflip operator
# ##############################
@curry
def mutate_bitflip(next_individual, expected=1):
    """ mutate and return an individual with a binary representation

    >>> from leap import core, binary_problems

    >>> original = Individual([1,1])

    >>> mutated = next(mutate_bitflip(iter([original])))

    :param individual: to be mutated
    :param expected: the *expected* number of mutations, on average
    :return: mutated individual
    """
    def flip(gene):
        if random.random() < probability:
            return (gene + 1) % 2
        else:
            return gene

    while True:
        # individual, pipe_args, pipe_kwargs = next(next_individual)
        individual = next(next_individual)

        # Given the average expected number of mutations, calculate the probability
        # for flipping each bit.  This calculation must be made each time given
        # that we may be dealing with dynamic lengths.
        probability = 1.0 / len(individual.genome) * expected

        individual.genome = [flip(gene) for gene in individual.genome]

        yield individual


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

@curry
def truncate(offspring, size, parents=None):
    """ return the `size` best individuals from the given population

        This defaults to (mu, lambda) if `parents` is not given.

        >>> from leap import core, ops, binary_problems
        >>> pop = [core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
        ...        core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
        ...        core.Individual([1, 1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
        ...        core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]

        We need to evaluate them to get their fitness to sort them for truncation.

        >>> pop = core.Individual.evaluate_population(pop)

        >>> truncated = truncate(pop, 2)

        TODO Do we want an optional context to over-ride the 'parents' parameter?

        :param offspring: offspring to truncate down to a smaller population
        :param size: is what to resize population to
        :param second_population: is optional parent population to include
                                  with population for downsizing
        :return: truncated population
    """
    if parents is not None:
        return toolz.itertoolz.topk(size, itertools.chain(offspring, parents))
    else:
        return toolz.itertoolz.topk(size, offspring)


def tournament(population, k=2):
    """ Selects the best individual from k individuals randomly selected from
        the given population

        >>> from leap import core, ops, binary_problems
        >>> pop = [core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
        ...        core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]

        We need to evaluate them to get their fitness to sort them for truncation.

        >>> pop = core.Individual.evaluate_population(pop)

        >>> best = tournament(pop)

        :param population: from which to select
        :param k: are randomly drawn from which to choose the best; by default this is 2 for binary tournament selection
        :return: the best of k individuals drawn from population
    """
    while True:
        choices = random.choices(population, k=k)
        best = max(choices)

        yield best


@curry
def naive_cyclic_selection_generator(population):
    """ Deterministically returns individuals, and repeats the same sequence
    when exhausted.

    TODO "cyclic_selection_generator" is a mouthful.  How about we rename this 
    to "unpool", since it's basically the opposite of pool()? --Siggy

    This is "naive" because it doesn't shuffle the population between complete
    tours to minimize bias.

    TODO implement non-naive version that shuffles population before first
    iteration and after every complete loop to minimize sample bias.

    >>> from leap import core, ops

    >>> pop = [core.Individual([0, 0]),
    ...        core.Individual([0, 1])]

    >>> cyclic_selector = ops.naive_cyclic_selection_generator(pop)

    :param population: from which to select
    :return: the next selected individual
    """
    itr = itertools.cycle(population)

    while True:
        yield next(itr)


@curry
def pool(next_individual, size):
    """ 'Sink' for creating `size` individuals from preceding pipeline source.

    Allows for "pooling" individuals to be processed by next pipeline
    operator.  Typically used to collect offspring from preceding set of
    selection and birth operators, but could also be used to, say, "pool"
    individuals to be passed to an EDA as a training set.

    >>> from leap import core, ops

    >>> pop = [core.Individual([0, 0]),
    ...        core.Individual([0, 1])]

    >>> cyclic_selector = ops.naive_cyclic_selection_generator(pop)

    >>> pool = ops.pool(cyclic_selector, 3)

    print(pool)
    [Individual([0, 0], None, None), Individual([0, 1], None, None), Individual([0, 0], None, None)]

    :param next_individual: generator for getting the next offspring
    :param size: how many kids we want
    :return: population of `size` offspring
    """
    return [next(next_individual) for _ in range(size)]
