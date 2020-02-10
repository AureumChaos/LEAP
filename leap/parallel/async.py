#!/usr/bin/env python3
"""
  This provides an asynchronous steady-state fitness evaluation pipeline
  operator.
"""
from toolz import curry

from dask.distributed import as_completed

from leap import core

from evaluate import evaluate


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


@curry
def eval_pool(next_individual, client, futures_iter, pool_size, num_births,
              context=core.context):
    """ Asynchronously evaluate `size` individuals

    This is similar to ops.pool() in that it's a "sink" for accumulating
    individuals by "pulling" individuals from upstream the pipeline via
    `next_individual`.  However, it's also like ops.evaluate() in that
    these individuals are concurrently evaluated via a map/reduce approach. We
    use dask to implement this evaluation mechanism.

    If an exception is thrown while evaluating an individual, NaN is assigned as
    its fitness, individual.is_viable is set to False, and the associated
    exception is assigned to individual.exception as a post mortem aid; also
    core.context['leap']['parallel']['non_viables'] count is incremented if you
    want to track the number of non-viable individuals (i.e., those that have
    an exception thrown during evaluation); just remember to reset that between
    runs if that variable has been updated.

    :param next_individual: iterator/generator for individual provider
    :param client: dask client through which we submit individuals to be evaluated
    :param futures_iter: dask distributed iterator to futures of individuals
        that are being evaluated or have finished evaluating
    :param pool_size: how many evaluated individuals to keep
    :param num_births: birth budget
    :param context: for storing count of non-viable individuals
    :return: the pool of evaluated individuals
    """
