""" Module for fundamental evolutionary operators.  You'll find many
traditional selection and reproduction strategies here.

"""

import abc
import itertools
import random

import numpy as np
import toolz
from toolz import curry

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
    def __call__(self, pop_generator):
        """
        The basic interface for a pipeline operator in LEAP.

        :param pop_generator: a list or generator of individuals to be operated upon
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


@curry
def uniform_crossover(next_individual, p_swap=0.5):
    """ Generator for recombining two individuals and passing them down the line.

    >>> from leap import core, binary_problems

    >>> first = Individual([0,0])
    >>> second = Individual([1,1])
    >>> i = iter([first, second])
    >>> result = uniform_crossover(i)

    >>> new_first = next(result)
    >>> new_second = next(result)

    :param next_individual: where we get the next individual
    :param p_swap: how likely are we to swap each pair of genes
    :return: two recombined individuals
    """
    def _uniform_crossover(ind1, ind2, p_swap):
        """ Recombination operator that can potentially swap any matching pair of
        genes between two individuals with some probability.

        It is assumed that ind1.genome and ind2.genome are lists of things.

        :param ind1: The first individual
        :param ind2: The second individual
        :param p_swap:

        :return: a copy of both individuals with individual.genome bits
                 swapped based on probability
        """
        if len(ind1.genome) != len(ind2.genome):
            # TODO what about variable length genomes?
            raise RuntimeError('genomes must be same length for uniform crossover')

        for i in range(len(ind1.genome)):
            if random.random() < p_swap:
                ind1.genome[i], ind2.genome[i] = ind2.genome[i], ind1.genome[i]

        return ind1, ind2

    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)

        child1, child2 = _uniform_crossover(parent1, parent2, p_swap)

        yield child1
        yield child2


@curry
def n_ary_crossover(next_individual, num_points=1):
    """ Do crossover between individuals between N crossover points.

    1 < n < genome length - 1

    We also assume that the passed in individuals are *clones* of parents.

    >>> from leap import core, binary_problems

    >>> first = Individual([0,0])
    >>> second = Individual([1,1])
    >>> i = iter([first, second])
    >>> result = n_ary_crossover(i)

    >>> new_first = next(result)
    >>> new_second = next(result)

    :param next_individual: where we get the next individual from the pipeline
    :param num_points: how many crossing points do we allow?
    :return: two recombined
    """
    def _pick_crossover_points(num_points, genome_size):
        """
        Randomly choose (without replacement) crossover points.
        """
        pp = list(range(0, genome_size))  # See De Jong, EC, pg 145

        xpts = [pp.pop(random.randrange(len(pp))) for i in range(num_points)]
        xpts.sort()
        xpts = [0] + xpts + [genome_size]  # Add start and end

        return xpts

    def _n_ary_crossover(child1, child2, num_points):
        # Sanity checks
        if len(child1.genome) != len(child2.genome):
            raise RuntimeError('Invalid length for n_ary_crossover')
        elif len(child1.genome) < num_points + 1:
            raise RuntimeError('Invalid number of crossover points for n_ary_crossover')

        children = [child1, child2]
        genome1 = child1.genome[0:0]  # empty sequence - maintain type
        genome2 = child2.genome[0:0]

        # Used to toggle which sub-sequence is copied between offspring
        src1, src2 = 0, 1

        # Pick crossover points
        xpts = _pick_crossover_points(num_points, len(child1.genome))

        for start, stop in toolz.itertoolz.sliding_window(2, xpts):
            genome1 += children[src1].genome[start:stop]
            genome2 += children[src2].genome[start:stop]

            # Now swap crossover direction
            src1, src2 = src2, src1

        child1.genome = genome1
        child2.genome = genome2

        return child1, child2

    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)

        child1, child2 = _n_ary_crossover(parent1, parent2, num_points)

        yield child1
        yield child2



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
def naive_cyclic_selection(population):
    """ Deterministically returns individuals, and repeats the same sequence
    when exhausted.

    This is "naive" because it doesn't shuffle the population between complete
    tours to minimize bias.

    >>> from leap import core, ops

    >>> pop = [core.Individual([0, 0]),
    ...        core.Individual([0, 1])]

    >>> cyclic_selector = ops.naive_cyclic_selection(pop)

    :param population: from which to select
    :return: the next selected individual
    """
    itr = itertools.cycle(population)

    while True:
        yield next(itr)


@curry
def cyclic_selection(population):
    """ Deterministically returns individuals in order, then shuffles the sequence, returns the individuals in that
    new order, and repeats this process.
    >>> from leap import core, ops

    >>> pop = [core.Individual([0, 0]),
    ...        core.Individual([0, 1])]

    >>> cyclic_selector = ops.cyclic_selection(pop)

    :param population: from which to select
    :return: the next selected individual
    """
    # this is essentially itertools.cycle() that just shuffles
    # the saved sequence between cycles.
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)
        for element in saved:
              yield element



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

    >>> cyclic_selector = ops.naive_cyclic_selection(pop)

    >>> pool = ops.pool(cyclic_selector, 3)

    print(pool)
    [Individual([0, 0], None, None), Individual([0, 1], None, None), Individual([0, 0], None, None)]

    :param next_individual: generator for getting the next offspring
    :param size: how many kids we want
    :return: population of `size` offspring
    """
    return [next(next_individual) for _ in range(size)]
