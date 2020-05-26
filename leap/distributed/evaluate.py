#!/usr/bin/env python#
"""
    contains common evaluate() used in sync.eval_pool and async.eval_pool
"""
import math
import time
import platform
import os
from toolz import curry

from dask.distributed import get_worker


from leap import core


@curry
def evaluate(individual, context=core.context):
    """ concurrently evaluate the given individual

    This is what's invoked on each dask worker to evaluate each individual.

    We log the start and end times for evaluation.

    An individual is viable if an exception is NOT thrown, else it is NOT a
    viable individual.  If not viable, we increment the context['leap'][
    'distributed']['non_viable'] count to track such instances.

    This function sets:

    individual.start_eval_time has the time() of when evaluation started.
    individual.stop_eval_time has the time() of when evaluation finished.
    individual.is_viable is True if viable, else False
    individual.exception will be assigned any raised exceptions
    individual.fitness will be NaN if not viable, else the calculated fitness
    individual.hostname is the name of the host on which this individual was
    evaluated
    individual.pid is the process ID associated with evaluating the individual

    :param individual: to be evaluated
    :return: evaluated individual
    """
    try:
        worker = get_worker()

        individual.start_eval_time = time.time()

        if hasattr(worker, 'logger'):
            worker.logger.debug(
                f'Worker {worker.id} started evaluating {individual!s}')
        individual.evaluate()
        individual.is_viable = True
    except Exception as e:
        # Set fitness to NaN to indicate we *tried* to evaluate the
        # individual; also save the associated exception so we can
        # (hopefully) figure out what went wrong.
        individual.fitness = math.nan
        individual.is_viable = False  # TODO maybe the NaN is sufficient?
        individual.exception = e

        # We track the number of such failures on the off chance that this
        # might be useful.
        context['leap']['distributed']['non_viable'] += 1

        if hasattr(worker, 'logger'):
            worker.logger.warning(
                f'Worker {worker.id}: {e} raised for {individual!s}')
    finally:
        individual.stop_eval_time = time.time()
        individual.hostname = platform.node()
        individual.pid = os.getpid()
        if hasattr(worker, 'logger'):
            worker.logger.debug(
                f'Worker {worker.id} evaluated {individual!s} in '
                f'{individual.stop_eval_time - individual.start_eval_time} '
                f'seconds')

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
