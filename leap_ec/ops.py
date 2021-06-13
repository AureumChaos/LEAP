"""Fundamental evolutionary operators.

This module provides many of the most important functions that we string
together to create EAs out of operator pipelines. You'll find many
traditional selection and reproduction strategies here, as well as components
for classic algorithms like island models and cooperative coevolution.

Representation-specific operators tend to reside within their own subpackages,
rather than here.  See for example :py:mod:`leap_ec.real_rep.ops` and
:py:mod:`leap_ec.binary_rep.ops`.
"""
import abc
import collections
from copy import copy
import csv
import itertools
from functools import wraps
import random
from statistics import mean
from typing import Iterator, List, Tuple, Callable

import numpy as np
import toolz
from toolz import curry

from leap_ec.global_vars import context
from leap_ec.individual import Individual


##############################
# Class Operator
##############################
class Operator(abc.ABC):
    """Abstract base class that documents the interface for operators in a
    LEAP pipeline.

    LEAP treats operators as functions of two arguments: the population,
    and a "context" `dict` that may be used in some algorithms to maintain
    some global state or parameters independent of the population.

    TODO The above description is outdated. --Siggy

    You can inherit from this class to define operators as classes.  Classes
    support operators that take extra arguments at construction time (such as
    a mutation rate) and maintain some internal private state, and they allow
    certain special patterns (such as multi-function operators).

    But inheriting from this class is optional.  LEAP can treat any
    `callable` object that takes two parameters as an operator.  You may
    define your custom operators as closures (which also allow for
    construction-time arguments and internal state), as simple functions (
    when no additional arguments are necessary), or as curried functions (
    i.e. with the help of `toolz.curry(...)`.

    """

    @abc.abstractmethod
    def __call__(self, pop_generator):
        """
        The basic interface for a pipeline operator in LEAP.

        :param pop_generator: a list or generator of individuals to be operated upon
        """
        pass


##############################
# Decorators for type checking
##############################
def iteriter_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives an iterator as its first argument, and that it
    returns an iterator.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs a list to an operator
    that expects an iterator, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(next_individual: Iterator, *args, **kwargs) -> Iterator:
        if not isinstance(next_individual, collections.abc.Iterator):
            if isinstance(next_individual, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(next_individual)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population iterator."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(next_individual)} as input, "
                    f"but expected an iterator.")

        result = f(next_individual, *args, **kwargs)

        if not isinstance(result, collections.abc.Iterator):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, but "
                f"expected an iterator.")

        return result

    return typecheck_f


def listlist_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives a list as its first argument, and that it returns
    a list.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs an iterator to an
    operator that expects a list, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(population: List, *args, **kwargs) -> List:
        if not isinstance(population, list):
            if isinstance(population, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(population)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population list."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(population)} as input, but "
                    f"expected a list.")

        result = f(population, *args, **kwargs)

        if not isinstance(result, list):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, but "
                f"expected a list.")

        return result

    return typecheck_f


def listiter_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives a list as its first argument, and that it returns
    an iterator.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs an iterator to an
    operator that expects a list, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(population: List, *args, **kwargs) -> Iterator:
        if not isinstance(population, list):
            if isinstance(population, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(population)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population list."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(population)} as input, but "
                    f"expected a list.")

        result = f(population, *args, **kwargs)

        if not isinstance(result, collections.abc.Iterator):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, but "
                f"expected an iterator.")

        return result

    return typecheck_f


def iterlist_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives an iterator as its first argument, and that it
    returns a list.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs a list to an operator
    that expects an iterator, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(next_individual: Iterator, *args, **kwargs) -> List:
        if not isinstance(next_individual, collections.abc.Iterator):
            if isinstance(next_individual, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(next_individual)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population iterator."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(next_individual)} as input, "
                    f"but expected an iterator.")

        result = f(next_individual, *args, **kwargs)

        if not isinstance(result, list):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, "
                f"but expected a list.")

        return result

    return typecheck_f


##############################
# evaluate operator
##############################
@curry
@iteriter_op
def evaluate(next_individual: Iterator) -> Iterator:
    """ Evaluate and returns the next individual in the pipeline

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.binary_rep.problems import MaxOnes

    We need to specify the decoder and problem so that evaluation is possible.

    >>> ind = Individual([1,1], decoder=IdentityDecoder(), problem=MaxOnes())

    >>> evaluated_ind = next(evaluate(iter([ind])))

    :param next_individual: iterator pointing to next individual to be evaluated

    :param kwargs: contains optional context state to pass down the pipeline
    in context dictionaries

    :return: the evaluated individual
    """
    while True:
        individual = next(next_individual)
        individual.evaluate()

        yield individual


##############################
# evaluate operator
##############################
@curry
@listlist_op
def grouped_evaluate(population: list, problem, max_individuals_per_chunk: int = None) -> list:
    """Evaluate the population by sending groups of multiple individuals to
    a fitness function so they can be evaluated simultaneously.

    This is useful, for example, as a way to evaluate individuals in parallel
    on a GPU."""
    if max_individuals_per_chunk is None:
        max_individuals_per_chunk = len(population)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    fitnesses = []
    for chunk in chunks(population, max_individuals_per_chunk):
        phenomes = [ ind.decode() for ind in chunk ]
        fit = problem.evaluate_multiple(phenomes)
        fitnesses.extend(fit)

    for fit, ind in zip(fitnesses, population):
        ind.fitness = fit

    return population


##############################
# const_evaluate operator
##############################
@curry
@listlist_op
def const_evaluate(population: List, value) -> List:
    """An evaluator that assigns a constant fitness to every individual.

    This ignores the `Problem` associated with each individual for the
    purpose of assigning a constant fitness.

    This is useful for algorithms that need to assign an arbitrary initial
    fitness value before using their normal evaluation method.  Some forms of
    cooperative coevolution are an example.
    """
    for ind in population:
        ind.fitness = value

    return population


##############################
# clone operator
##############################
@curry
@iteriter_op
def clone(next_individual: Iterator) -> Iterator:
    """ clones and returns the next individual in the pipeline

    >>> from leap_ec.individual import Individual

    Create a common decoder and problem for individuals.

    >>> original = Individual([1,1])

    >>> cloned_generator = clone(iter([original]))

    :param next_individual: iterator for next individual to be cloned
    :return: copy of next_individual
    """

    while True:
        individual = next(next_individual)

        yield individual.clone()


##############################
# Function uniform_crossover
##############################
@curry
@iteriter_op
def uniform_crossover(next_individual: Iterator,
                      p_swap: float = 0.2, p_xover: float = 1.0) -> Iterator:
    """Parameterized uniform crossover iterates through two parents' genomes
    and swaps each of their genes with the given probability.

    In a classic paper, De Jong and Spears showed that this operator works
    particularly well when the swap probability `p_swap` is set to about 0.2.  LEAP
    thus uses this value as its default.

        De Jong, Kenneth A., and W. Spears. "On the virtues of parameterized uniform crossover."
        *Proceedings of the 4th international conference on genetic algorithms.* Morgan Kaufmann Publishers, 1991.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import uniform_crossover

    >>> first = Individual([0,0])
    >>> second = Individual([1,1])
    >>> i = iter([first, second])
    >>> result = uniform_crossover(i)

    >>> new_first = next(result)
    >>> new_second = next(result)

    The probability can be tuned via the `p_swap` parameter:

    >>> result = uniform_crossover(i, p_swap=0.1)

    :param next_individual: where we get the next individual
    :param p_swap: how likely are we to swap each pair of genes when crossover
        is performed
    :param float p_xover: the probability that crossover is performed in the
        first place
    :return: two recombined individuals (with probability p_xover), or two
        unmodified individuals (with probability 1 - p_xover)
    """

    def _uniform_crossover(ind1, ind2, p_swap):
        """ Recombination operator that can potentially swap any matching pair of
        genes between two individuals with some probability.

        It is assumed that ind1.genome and ind2.genome are lists of things.

        :param ind1: The first individual
        :param ind2: The second individual
        :param p_swap: how likely are we to swap each pair of genes when crossover
            is performed

        :return: a copy of both individuals with individual.genome bits
                 swapped based on probability
        """
        if len(ind1.genome) != len(ind2.genome):
            # TODO what about variable length genomes?
            raise RuntimeError(
                'genomes must be same length for uniform crossover')

        for i in range(len(ind1.genome)):
            if random.random() < p_swap:
                ind1.genome[i], ind2.genome[i] = ind2.genome[i], ind1.genome[i]

        return ind1, ind2

    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)
        # Return the parents unmodified if we're not performing crossover
        if np.random.uniform() > p_xover:
            yield parent1
            yield parent2
        else:  # Else do crossover
            child1, child2 = _uniform_crossover(parent1, parent2, p_swap)

            # Invalidate fitness since the genomes have changed
            child1.fitness = child2.fitness = None

            yield child1
            yield child2


##############################
# Function n_ary_crossover
##############################
@curry
@iteriter_op
def n_ary_crossover(next_individual: Iterator,
                    num_points: int = 2,
                    p=1.0) -> Iterator:
    """ Do crossover between individuals between N crossover points.

    1 < n < genome length - 1

    We also assume that the passed in individuals are *clones* of parents.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import n_ary_crossover

    >>> first = Individual([0,0])
    >>> second = Individual([1,1])
    >>> i = iter([first, second])
    >>> result = n_ary_crossover(i)

    >>> new_first = next(result)
    >>> new_second = next(result)

    :param next_individual: where we get the next individual from the pipeline
    :param num_points: how many crossing points do we use?  Defaults to 2, since
        2-point crossover has been shown to be the least disruptive choice for
        this value.
    :param p: the probability that crossover is performed.
    :return: two recombined
    """

    def _pick_crossover_points(num_points, genome_size):
        """
        Randomly choose (without replacement) crossover points.
        """
        pp = list(range(0, genome_size))  # See De Jong, EC, pg 145

        xpts = sorted([pp.pop(random.randrange(len(pp)))
                       for i in range(num_points)])
        xpts = [0] + xpts + [genome_size]  # Add start and end

        return xpts

    def _n_ary_crossover(child1, child2, num_points):
        if len(child1.genome) < num_points or \
                len(child2.genome) < num_points:
            raise RuntimeError(
                'Invalid number of crossover points for n_ary_crossover')

        children = [child1, child2]
        genome1 = child1.genome[0:0]  # empty test_sequence - maintain type
        genome2 = child2.genome[0:0]

        # Used to toggle which sub-test_sequence is copied between offspring
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

        # Return the parents unmodified if we're not performing crossover
        if np.random.uniform() > p:
            yield parent1
            yield parent2
        else:  # Else cross them over
            child1, child2 = _n_ary_crossover(parent1, parent2, num_points)
            yield child1
            yield child2


##############################
# Function truncation_selection
##############################
@curry
@listlist_op
def truncation_selection(offspring: List, size: int,
                         parents: List = None,
                         key = None) -> List:
    """ return the `size` best individuals from the given population

        This defaults to (mu, lambda) if `parents` is not given.

        >>> from leap_ec.individual import Individual
        >>> from leap_ec.decoder import IdentityDecoder
        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.ops import truncation_selection

        >>> pop = [Individual([0, 0, 0], decoder=IdentityDecoder(), problem=MaxOnes()),
        ...        Individual([0, 0, 1], decoder=IdentityDecoder(), problem=MaxOnes()),
        ...        Individual([1, 1, 0], decoder=IdentityDecoder(), problem=MaxOnes()),
        ...        Individual([1, 1, 1], decoder=IdentityDecoder(), problem=MaxOnes())]

        We need to evaluate them to get their fitness to sort them for
        truncation.

        >>> pop = Individual.evaluate_population(pop)

        >>> truncated = truncation_selection(pop, 2)

        TODO Do we want an optional context to over-ride the 'parents' parameter?

        :param offspring: offspring to truncate down to a smaller population
        :param size: is what to resize population to
        :param second_population: is optional parent population to include
                                  with population for downsizing
        :return: truncated population
    """
    if key:
        if parents is not None:
            return list(toolz.itertoolz.topk(
                size, itertools.chain(offspring, parents), key=key))
        else:
            return list(toolz.itertoolz.topk(size, offspring, key=key))
    else:
        if parents is not None:
            return list(toolz.itertoolz.topk(
                size, itertools.chain(offspring, parents)))
        else:
            return list(toolz.itertoolz.topk(size, offspring))


##############################
# Function elitist_survival
##############################
@curry
@listlist_op
def elitist_survival(offspring: List, parents: List, k: int = 1, key = None) -> List:
    """ This allows k best parents to compete with the offspring.

        >>> from leap_ec.individual import Individual
        >>> from leap_ec.decoder import IdentityDecoder as ID
        >>> from leap_ec.binary_rep.problems import MaxOnes

        First, let's make a "pretend" population of parents using the MaxOnes
        problem.

        >>> pretend_parents = [Individual([0, 0, 0], decoder=ID(), problem=MaxOnes()), Individual([1, 1, 1], decoder=ID(), problem=MaxOnes())]

        Then a "pretend" population of offspring. (Pretend in that we're
        pretending that the offspring came from the parents.)

        >>> pretend_offspring = [Individual([0, 0, 0], decoder=ID(), problem=MaxOnes()), Individual([1, 1, 0], decoder=ID(), problem=MaxOnes()), Individual([1, 0, 1], decoder=ID(), problem=MaxOnes()), Individual([0, 1, 1], decoder=ID(), problem=MaxOnes()), Individual([0, 0, 1], decoder=ID(), problem=MaxOnes())]

        We need to evaluate them to get their fitness to sort them for
        elitist_survival.

        >>> pretend_parents = Individual.evaluate_population(pretend_parents)
        >>> pretend_offspring = Individual.evaluate_population(pretend_offspring)

        This will take the best parent, which has [1,1,1], and replace the
        worst offspring, which has [0,0,0] (because this is the MaxOnes problem)
        >>> survivors = elitist_survival(pretend_offspring, pretend_parents)

        >>> assert pretend_parents[1] in survivors # yep, best parent is there
        >>> assert pretend_offspring[0] not in survivors # worst guy isn't

        We orginally ordered 5 offspring, so that's what we better have.
        >>> assert len(survivors) == 5

        Please note that the literature has a number of variations of elitism
        and other forms of overlapping generations.  For example, this may be a
        good starting point:

        De Jong, Kenneth A., and Jayshree Sarma. "Generation gaps revisited."
        In Foundations of genetic algorithms, vol. 2, pp. 19-28. Elsevier, 1993.

    :param offspring: list of created offpring, probably from pool()
    :param parents: list of parents, usually the ones that offspring came from
    :param k: how many elites from parents to keep?
    :param key: optional key criteria for selecting; e.g., can be used to impose
        parsimony pressure
    :return: surviving population, which will be offspring with offspring
        replaced by any superior parent elites
    """
    # We save this because we're going to truncate back down to this number
    # for the final survivors
    original_num_offspring = len(offspring)

    # Append the requested number of best parents to the offspring.
    if key:
        elites = list(toolz.itertoolz.topk(k, parents, key=key))
    else:
        elites = list(toolz.itertoolz.topk(k, parents))
    offspring.extend(elites)

    # Now return the offspring (plus possibly an elite) truncating the least
    # fit individual.
    if key:
        return list(toolz.itertoolz.topk(original_num_offspring, offspring, key))
    else:
        return list(toolz.itertoolz.topk(original_num_offspring, offspring))


##############################
# Function tournament_selection
##############################
@curry
@listiter_op
def tournament_selection(population: list, k: int = 2, key = None, select_worst: bool=False, indices = None) -> Iterator:
    """Returns an opertaor that selects the best individual from k individuals randomly selected from
        the given population.

        Like other selection operators, this assumes that if one individual is "greater than" another, then it is 
        "better than" the other.  Whether this indicates maximization or minimization isn't handled here: the
        `Individual` class determines the semantics of its "greater than" operator.

        :param population: the population to select from.  Should be a list, not an iterator.
        :param int k: number of contestants in the tournament.  k=2 does binary tournament
            selection, which approximates linear ranking selection in the expectation.  Higher
            values of k yield greedier selection strategies—k=3, for instance, is equal to 
            quadratic ranking selection in the expectation.
        :param key: an optional function that computes keys to sort over.  Defaults to None,
            in which case Individuals are compared directly.
        :param bool select_worst: if True, select the worst individual from the tournament instead
            of the best.
        :param list indices: an optional 

        >>> from leap_ec import Individual
        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.ops import tournament_selection

        >>> pop = [Individual([0, 0, 0], problem=MaxOnes()),
        ...        Individual([0, 0, 1], problem=MaxOnes())]
        >>> pop = Individual.evaluate_population(pop)
        >>> best = tournament_selection(pop)

        :param population: from which to select
        :param k: are randomly drawn from which to choose the best; by
            default this is 2 for binary tournament selection
        :param key: optional max() key

        :return: the best of k individuals drawn from population
    """
    assert((indices is None) or (isinstance(indices, list))), f"Only a list should be passed to tournament_selection() for indices, but received {indices}."

    while True:
        choices_idx = random.choices(range(len(population)), k=k)
        judge = min if select_worst else max
        if key:
            best_idx = judge(choices_idx, key=lambda x: key(population[x]))
        else:
            best_idx = judge(choices_idx, key=lambda x: population[x])

        if indices is not None:
            indices.clear()  # Nuke whatever is in there
            indices.append(best_idx)  # Add the index of the individual we're about to return

        yield population[best_idx]


##############################
# Function insertion_selection
##############################
@curry
@listlist_op
def insertion_selection(offspring: List, parents: List, key = None) -> List:
    """ do exclusive selection between offspring and parents

    This is typically used for Ken De Jong's EV algorithm for survival
    selection.  Each offspring is deterministically selected and a random
    parent is selected; if the offspring wins, then it replaces the parent.

    Note that we make a _copy_ of the parents and have the offspring compete
    with the parent copies so that users can optionally preserve the original
    parents.  You may wish to do that, for example, if you want to analyze the
    composition of the original parents and the modified copy.

    :param offspring: population to select from
    :param parents: parents that are copied and which the copies are
           potentially updated with better offspring
    :param key: optional key for determining max() by other criteria such as
        for parsimony pressure
    :return: the updated parent population
    """
    copied_parents = copy(parents)
    for child in offspring:
        selected_parent_index = random.randrange(len(copied_parents))
        if key:
            copied_parents[selected_parent_index] = max(child,
                                                        copied_parents[
                                                            selected_parent_index],
                                                        key=key)
        else:
            copied_parents[selected_parent_index] = max(child,
                                                        copied_parents[
                                                            selected_parent_index])

        return copied_parents


##############################
# Function naive_cyclic_selection
##############################
@curry
@listiter_op
def naive_cyclic_selection(population: List) -> Iterator:
    """ Deterministically returns individuals, and repeats the same test_sequence
    when exhausted.

    This is "naive" because it doesn't shuffle the population between complete
    tours to minimize bias.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import naive_cyclic_selection

    >>> pop = [Individual([0, 0]),
    ...        Individual([0, 1])]

    >>> cyclic_selector = naive_cyclic_selection(pop)

    :param population: from which to select
    :return: the next selected individual
    """
    itr = itertools.cycle(population)

    while True:
        yield next(itr)


##############################
# Function cyclic_selection
##############################
@curry
@listiter_op
def cyclic_selection(population: List) -> Iterator:
    """ Deterministically returns individuals in order, then shuffles the
    test_sequence, returns the individuals in that new order, and repeats this
    process.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import cyclic_selection

    >>> pop = [Individual([0, 0]),
    ...        Individual([0, 1])]

    >>> cyclic_selector = cyclic_selection(pop)

    :param population: from which to select
    :return: the next selected individual
    """
    # this is essentially itertools.cycle() that just shuffles
    # the saved test_sequence between cycles.
    saved = []
    for individual in population:
        yield individual
        saved.append(individual)
    while saved:
        # randomize the test_sequence between cycles to remove this source of sample
        # bias
        random.shuffle(saved)
        for individual in saved:
            yield individual


##############################
# Function random_selection
##############################
@listiter_op
def random_selection(population: List) -> Iterator:
    """ return a uniformly randomly selected individual from the population

    :param population: from which to select
    :return: a uniformly selected individual
    """
    while True:
        yield random.choice(population)


##############################
# Function pool
##############################
@curry
@iterlist_op
def pool(next_individual: Iterator, size: int) -> List:
    """ 'Sink' for creating `size` individuals from preceding pipeline source.

    Allows for "pooling" individuals to be processed by next pipeline
    operator.  Typically used to collect offspring from preceding set of
    selection and birth operators, but could also be used to, say, "pool"
    individuals to be passed to an EDA as a training set.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import naive_cyclic_selection

    >>> pop = [Individual([0, 0]),
    ...        Individual([0, 1])]

    >>> cyclic_selector = naive_cyclic_selection(pop)

    >>> pool = pool(cyclic_selector, 3)

    print(pool)
    [Individual([0, 0], None, None), Individual([0, 1], None, None), Individual([0, 0], None, None)]

    :param next_individual: generator for getting the next offspring
    :param size: how many kids we want
    :return: population of `size` offspring
    """
    return [next(next_individual) for _ in range(size)]


##############################
# Function migrate
##############################
def migrate(topology, emigrant_selector,
            replacement_selector, migration_gap,
            customs_stamp=lambda x, _: x,
            context=context):
    num_islands = topology.number_of_nodes()

    # We wrap a closure around some persistent state to keep trag of
    # immigrants as the move between populations
    immigrants = [[] for i in range(num_islands)]

    @listlist_op
    def do_migrate(population: List) -> List:
        current_subpop = context['leap']['current_subpopulation']

        # Immigration
        for imm in immigrants[current_subpop]:
            # Do island-specific transformation
            # For example, this callback might update the individuals 'problem'
            # field to point to a new fitness function for the island, and
            # re-evalute its fitness.
            imm = customs_stamp(imm, current_subpop)
            # Compete for a place in the new population
            contestant = next(replacement_selector(population))
            if imm > contestant:
                # FIXME This is fishy!  What if there are two copies of
                # contestant?  What if contestant.__eq()__ is not properly
                # implemented?
                population.remove(contestant)
                population.append(imm)

        immigrants[current_subpop] = []

        # Emigration
        if context['leap']['generation'] % migration_gap == 0:
            # Choose an emigrant individual
            sponsor = next(emigrant_selector(population))
            # Clone it and copy fitness
            emi = next(emigrant_selector(population)).clone()
            emi.fitness = sponsor.fitness
            neighbors = topology.neighbors(
                current_subpop)  # Get neighboring islands
            # Randomly select a neighboring island
            dest = random.choice(list(neighbors))
            # Add the emigrant to its immigration list
            immigrants[dest].append(emi)
            # FIXME In a heterogeneous island model, we also need to
            # set the emigrant's decoder and/or problem to match the
            # new islan'ds decoder and/or problem.

        return population

    return do_migrate


##############################
# Class coop_evaluate
##############################
def concat_combine(collaborators):
    """Combine a list of individuals by concatenating their genomes.

    You can choose whether this or some other function is used for combining
    collaborators by passing it into the `CooperativeEvaluate` constructor. """
    # Clone one of the evaluators so we can use its problem and decoder later
    combined_ind = collaborators[0].clone()

    genomes = [ind.genome for ind in collaborators]
    combined_ind.genome = list(itertools.chain(*genomes))  # Concatenate
    return combined_ind


class CooperativeEvaluate(Operator):
    """A simple, non-parallel implementation of cooperative coevolutionary
    fitness evaluation.

    :param context: the algorithm's state context.  Used to access
        subpopulation information.
    """

    def __init__(self, num_trials, collaborator_selector,
                 log_stream=None, combine=concat_combine,context=context):
        self.context = context
        self.num_trials = num_trials
        self.collaborator_selector = collaborator_selector
        self.combine = combine

        # Set up the CSV writier
        if log_stream is not None:
            self.log_writer = csv.DictWriter(
                log_stream,
                fieldnames=[
                    'generation',
                    'subpopulation',
                    'individual_type',
                    'collaborator_subpopulation',
                    'genome',
                    'fitness'])
            # We print the header at construction time
            self.log_writer.writeheader()

    def __call__(self, next_individual: Iterator) -> Iterator:
        """Execute the evaluation operator on a subpopulation."""
        while True:
            current_ind = next(next_individual)

            # Pull references to all subpopulations from the context object
            subpopulations = self.context['leap']['subpopulations']
            current_subpop = self.context['leap']['current_subpopulation']

            # Create iterators that select individuals from each subpopulation
            selectors = [self.collaborator_selector(
                subpop) for subpop in subpopulations]

            # Choose collaborators and evaulate
            fitnesses = []
            for i in range(self.num_trials):
                collaborators = self._choose_collaborators(
                    current_ind, subpopulations, current_subpop, selectors)
                combined_ind = self.combine(collaborators)
                fitness = combined_ind.evaluate()
                # Optionally write out data about the collaborations
                if self.log_writer is not None:
                    self._log_trial(
                        self.log_writer,
                        collaborators,
                        combined_ind,
                        i,
                        context=self.context)

                fitnesses.append(fitness)

            current_ind.fitness = mean(fitnesses)

            yield current_ind

    @staticmethod
    def _choose_collaborators(current_ind, subpopulations,
                              current_subpop, selectors):
        """Choose collaborators from the subpopulations."""
        collaborators = []
        for i in range(len(subpopulations)):
            if i != current_subpop:
                # Select a fellow collaborator from the other subpopulations
                ind = next(selectors[i])
                # Make sure we actually got something with a genome back
                assert (hasattr(ind, 'genome'))
                collaborators.append(ind)
            else:
                # Stick this subpop's individual in as-is
                collaborators.append(current_ind)

        assert (len(collaborators) == len(subpopulations))
        return collaborators

    @staticmethod
    def _log_trial(writer, collaborators, combined_ind, trial_id,
                   context=context):
        """Record information about a batch of collaborators to a CSV writer."""
        for i, collab in enumerate(collaborators):
            writer.writerow({'generation'                : context['leap']['generation'],
                             'subpopulation'             : context['leap']['current_subpopulation'],
                             'individual_type'           : 'Collaborator',
                             'collaborator_subpopulation': i,
                             'genome'                    : collab.genome,
                             'fitness'                   : collab.fitness})

        writer.writerow({'generation'                : context['leap']['generation'],
                         'subpopulation'             : context['leap']['current_subpopulation'],
                         'individual_type'           : 'Combined Individual',
                         'collaborator_subpopulation': None,
                         'genome'                    : combined_ind.genome,
                         'fitness'                   : combined_ind.fitness})


##############################
# function compute_expected_probability
##############################
def compute_expected_probability(expected_num_mutations: float,
                                 individual_genome: List) \
        -> float:
    """ Computed the probability of mutation based on the desired average
    expected mutation and genome length.

    :param expected_num_mutations: times individual is to be mutated on average
    :param individual_genome: genome for which to compute the probability
    :return: the corresponding probability of mutation
    """
    return 1.0 / len(individual_genome) * expected_num_mutations
