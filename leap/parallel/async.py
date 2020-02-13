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
from toolz import curry

from dask.distributed import as_completed

from leap import core

from evaluate import evaluate

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
    :param second:
    :param bag: of already evaluated individuals
    :param index: of individual in bag to be compared against
    :return: None
    """
    if new_individual > bag[index]:
        logger.debug('Replaced %s with %s', new_individual, bag[index])
        bag[index] = new_individual
    else:
        logger.debug('%s < %s, so the latter stays in bag', new_individual,
                     bag[index])


def insert_into_bag(indivdidual, bag):
    """ Insert the given individual into the bag of evaluated individuals.

    Randomly select an individual in the bag, and the `individual` will
    replace the selected individual iff it has a better fitness.

    :param indivdidual: that was just evaluated
    :param bag: of already evaluated individuals
    :return: None
    """
    rand_index = random.randrange(len(bag))
    replace_if(indivdidual, rand_index, bag)



def greedy_insert_into_bag(individual, bag):
    """ Insert the given individual into the bag of evaluated individuals.

    This is greedy because we always compare the new `individual` with the
    current weakest in the bag.

    :param individual: that was just evaluated
    :param bag: of already evaluated individuals
    :return: None
    """
    # From https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
    index_min = min(range(len(bag)), key=bag.__getitem__)
    replace_if(individual, bag, index_min)
