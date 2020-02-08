#!/usr/bin/env python3
"""
  This provides a synchronous fitness evaluation pipeline operator.
"""
impor math
from toolz import curry

from leap import core

@curry
def sync_eval_pool(next_individual, client, size, context=core.context):
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
    def evaluate(individual):
        """ concurrently evaluate the given individual

        :param individual: to be evaluated
        :return: evaluated individual
        """
        try:
            individual.evaluate()
        except Exception as e:
            individual.fitness = math.nan

        return individual

    # First, accumulate individuals to be evaluated
    unevaluated_offspring = [next(next_individual) for _ in range(size)]