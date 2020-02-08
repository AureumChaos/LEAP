#!/usr/bin/env python3
"""
  This provides a synchronous fitness evaluation pipeline operator.
"""
import math
from toolz import curry

from leap import core


def evaluate(individual):
    """ concurrently evaluate the given individual

    This is what's invoked on each dask worker.

    :param individual: to be evaluated
    :return: evaluated individual
    """
    try:
        individual.evaluate()
    except Exception as e:
        # Set fitness to NaN to indicate we *tried* to evaluate the
        # individual; also save the associated exception so we can
        # (hopefully) figure out what went wrong.
        individual.fitness = math.nan
        individual.exception = e

        # We track the number of such failures on the off chance that this
        # might be useful.
        context['leap']['parallel']['non_viable'] += 1

    return individual


def eval_population(population, client, context=core.context):
    """ Concurrently evaluate all the individuals in the given population

    :param population: to be evaluated
    :param client: dask client
    :param context: for storing count of non-viable individuals
    :return: evaluated population
    """
    # farm out population to worker nodes for evaluation
    worker_futures = self.client.map(evaluate, population)

    # now gather all the *completed* evaluations; note that some of the
    # evaluations may complete much earlier than others, which means those
    # related computational resources will idle until the last offspring is
    # evaluated.  If this is a problem, please consider using async_eval_pool,
    # instead.
    evaluated_individuals = client.gather(worker_futures)

    return evaluated_individuals


@curry
def eval_pool(next_individual, client, size, context=core.context):
    """ concurrently evaluate `size` individuals

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
    :param size: how many individuals to evaluate simultaneously.
    :param context: for storing count of non-viable individuals
    :return: the pool of evaluated individuals
    """
    # First, accumulate individuals to be evaluated
    unevaluated_offspring = [next(next_individual) for _ in range(size)]

    evaluated_offspring = eval_population(unevaluated_offspring, client,
                                          context)

    return evaluated_offspring
