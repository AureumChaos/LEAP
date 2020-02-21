#!/usr/bin/env python#
"""
    contains common evaluate() used in sync.eval_pool and async.eval_pool
"""
import math
from toolz import curry

from leap import core


@curry
def evaluate(individual, context=core.context):
    """ concurrently evaluate the given individual

    This is what's invoked on each dask worker.

    :param individual: to be evaluated
    :return: evaluated individual
    """
    try:
        individual.evaluate()
        individual.is_viable = True
    except Exception as e:
        # Set fitness to NaN to indicate we *tried* to evaluate the
        # individual; also save the associated exception so we can
        # (hopefully) figure out what went wrong.
        individual.fitness = math.nan
        individual.is_viable = False # TODO maybe the NaN is sufficient?
        individual.exception = e

        # We track the number of such failures on the off chance that this
        # might be useful.
        context['leap']['distributed']['non_viable'] += 1

    return individual


def is_viable(individual):
    """
    evaluate.evaluate() will set an individual's fitness to NaN and the
    attributes `is_viable` to False, and will assign any exception triggered
    during the individuals evaluation to `exception`.  This just checks the
    individual's `is_viable`; if it doesn't have one, this assumes it is viable.

    :param individual: to be checked if viable
    :return: True if individual is viable
    """
    if hasattr(individual, 'is_viable'):
        return individual.is_viable
    else:
        return True
