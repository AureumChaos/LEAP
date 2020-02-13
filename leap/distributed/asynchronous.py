#!/usr/bin/env python3
"""
  This provides an asynchronous steady-state fitness evaluation pipeline
  operator.

  A common feature here is a "bag" of evaluated individuals that is
  asynchronously updated via dask.  (We would use "pool instead of "bag",
  but "pool" is already in use as ops.pool().)

"""
import random
import logging
import math
from toolz import curry

from dask.distributed import Client, as_completed

from leap import core
from leap import util

from .evaluate import evaluate

# Create unique logger for this namespace
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def eval_population(population, client, context=core.context):
    """ Concurrently evaluate all the individuals in the given population

    :param population: to be evaluated
    :param client: dask client
    :param context: for storing count of non-viable individuals
    :return: dask distributed iterator for futures
    """
    # farm out population to worker nodes for evaluation
    worker_futures = client.map(evaluate(context=context), population)

    # We'll need this later to catch eval tasks as they complete, and to
    # submit new tasks.
    return as_completed(worker_futures)


def replace_if(new_individual, bag, index):
    """ Convenience function for possibly replacing bag[index] individual
    with new_individual depending on which has higher fitness.

    :param new_individual: is a newly evaluated individual
    :param bag: of already evaluated individuals
    :param index: of individual in bag to be compared against
    :return: None
    """
    # If individual in the bag has a NaN, that means it's non-viable, and will
    # *always lose*, even if the new individual is also a NaN.  (Thus assuring
    # there is churn in the bag.
    if bag[index].fitness == math.nan:
        logger.debug('Replacing %s for %s due to non-viable individual', new_individual, bag[index])
        bag[index] = new_individual
    elif new_individual > bag[index]:
        logger.debug('Replaced %s with %s', new_individual, bag[index])
        bag[index] = new_individual
    else:
        logger.debug('%s < %s, so the latter stays in bag', new_individual,
                     bag[index])


def insert_into_bag(indivdidual, bag, max_size):
    """ Insert the given individual into the bag of evaluated individuals.

    Randomly select an individual in the bag, and the `individual` will
    replace the selected individual iff it has a better fitness.

    Just insert individuals if the bag isn't at capacity yet

    :param indivdidual: that was just evaluated
    :param bag: of already evaluated individuals
    :param max_size: of the bag
    :return: None
    """
    if len(bag) < max_size:
        logger.debug('bag not at capacity, so just inserting')
        bag.append(indivdidual)
    else:
        rand_index = random.randrange(len(bag))
        replace_if(indivdidual, bag, rand_index)



def greedy_insert_into_bag(individual, bag, max_size):
    """ Insert the given individual into the bag of evaluated individuals.

    This is greedy because we always compare the new `individual` with the
    current weakest in the bag.

    Just insert individuals if the bag isn't at capacity yet

    :param individual: that was just evaluated
    :param bag: of already evaluated individuals
    :return: None
    """
    if len(bag) < max_size:
        logger.debug('bag not at capacity, so just inserting')
        bag.append(individual)
    else:
        # From https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
        index_min = min(range(len(bag)), key=bag.__getitem__)
        replace_if(individual, bag, index_min)


def is_viable(individual):
    """
    evaluate.evaluate() will set an individual's fitness to NaN and the
    attributes `is_viable` to False, and will assign any exception triggered
    during the individuals evaluation to `exception`.  This just checks the
    individual's `is_viable`; if it doesn't have one, this assumes it is viable.

    :param individual:
    :return: True if individual is viable
    """
    if hasattr(individual, 'is_viable'):
        return individual.is_viable
    else
        return True


def steady_state(client, births, init_pop_size, bag_size, individual_cls, initializer, decoder, problem, pipeline, count_nonviable=False):
    """ Implements an asynchronous steady-state EA

    :param client: Dask client that should already be set-up
    :param births: how many births are we allowing?
    :param init_pop_size: size of initial population
    :param bag_size: how large should the bag be?
    :param individual_cls: class prototype for Individual to be used
    :param initializer: how to initialize genomes for the first random population
    :param decoder: to to translate the genome into something the problem can understand
    :param problem: to be solved
    :param pipeline: for creating new offspring from the bag
    :param count_nonviable: True if we want to count non-viable individuals towards the birth budget
    :return: the bag containing the final individuals
    """
    initial_population = individual_cls.create_population(init_pop_size,
                                                initialize=initializer,
                                                decoder=decoder, problem=problem)

    # fan out the entire initial population to dask workers
    as_completed_iter = eval_population(initial_population, client=client)

    # This is where we'll be putting evaluated individuals
    bag = []

    # Bookeeping for tracking the number of births
    birth_count = util.inc_births(core.context, start=len(initial_population))

    for i, evaluated_future in enumerate(as_completed_iter):

        evaluated = evaluated_future.result()

        logger.debug('%d evaluated: %s %s', i, str(evaluated.genome), str(evaluated.fitness))

        if not count_nonviable and not is_viable(eval_population()):
            # If we don't want non-viable individuals to count towards the
            # birth budget, then we need to decrement the birth count that was
            # incremented when it was created.
            birth_count.do_decrement()

        asynchronous.greedy_insert_into_bag(evaluated, bag, bag_size)

        if birth_count.births() < births:
            # Only create offspring if we have the budget for one
            offspring = toolz.pipe(bag,*pipeline)

            logger.debug('created offspring: ')
            [logger.debug('%s', str(o.genome)), for o in offspring]

            # Now asynchronously submit to dask
            as_completed_iter.add(
                client.map(evaluate.evaluate(context=core.context),
                              offspring))

            birth_count.do_increment(len(offspring))

    return bag
