#!/usr/bin/env python3
"""
  This provides an asynchronous steady-state fitness evaluation pipeline
  operator.

  A common feature here is a population of evaluated individuals that is
  asynchronously updated via dask.

"""
import random
import logging
import toolz

import distributed

from leap_ec.global_vars import context
from leap_ec import util

from .evaluate import evaluate, is_viable
from .individual import DistributedIndividual

# Create unique logger for this namespace
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

##############################
# function eval_population
##############################
def eval_population(population, client, context=context):
    """ Concurrently evaluate all the individuals in the given population

    :param population: to be evaluated
    :param client: dask client
    :param context: for storing count of non-viable individuals
    :return: dask distrib iterator for futures
    """
    # farm out population to worker nodes for evaluation
    worker_futures = client.map(evaluate(context=context), population,
                                pure=False)

    # We'll need this later to catch eval tasks as they complete, and to
    # submit new tasks.
    return distributed.as_completed(worker_futures)


##############################
# function replace_if
##############################
def replace_if(new_individual, pop, index):
    """ Convenience function for possibly replacing pop[index] individual
    with new_individual depending on which has higher fitness.

    :param new_individual: is a newly evaluated individual
    :param pop: of already evaluated individuals
    :param index: of individual in pop to be compared against
    :return: None
    """
    # If individual in the pop is not viable, it will *always lose*, even if
    # the new individual is also not viable.  (Thus assuring there is churn
    # in the pop.
    if not is_viable(pop[index]):
        logger.debug('Replacing %s for %s due to non-viable individual',
                     new_individual, pop[index])
        pop[index] = new_individual
    elif new_individual > pop[index]:
        logger.debug('Replaced %s with %s', pop[index], new_individual)
        pop[index] = new_individual
    else:
        logger.debug('%s < %s, so the latter stays in pop', new_individual,
                     pop[index])


##############################
# function tournament_insert_into_pop
##############################
def tournament_insert_into_pop(individual, pop, max_size):
    """ Insert the given individual into the pop of evaluated individuals.

    Randomly select an individual in the pop, and the `individual` will
    replace the selected individual iff it has a better fitness. This is
    essentially binary tournament selection.

    Just insert individuals if the pop isn't at capacity yet

    TODO as with tournament selection, we should have an optional `k` to
    specify the tournament size. However, we have to be mindful that this is
    already k=2, so we would have to draw k-1 individuals from the population
    for comparison.

    :param individual: that was just evaluated
    :param pop: of already evaluated individuals
    :param max_size: of the pop
    :return: None
    """
    if len(pop) < max_size:
        logger.debug('pop not at capacity, so just inserting')
        pop.append(individual)
    else:
        rand_index = random.randrange(len(pop))
        replace_if(individual, pop, rand_index)


##############################
# function greedy_insert_into_pop
##############################
def greedy_insert_into_pop(individual, pop, max_size):
    """ Insert the given individual into the pop of evaluated individuals.

    This is greedy because we always compare the new `individual` with the
    current weakest in the pop.  This is similar to tournament selection.

    Just insert individuals if the pop isn't at capacity yet

    :param individual: that was just evaluated
    :param pop: of already evaluated individuals
    :return: None
    """
    if len(pop) < max_size:
        logger.debug('pop not at capacity, so just inserting')
        pop.append(individual)
    else:
        # From https://stackoverflow.com/questions/2474015/getting-the-index
        # -of-the-returned-max-or-min-item-using-max-min-on-a-list
        index_min = min(range(len(pop)), key=pop.__getitem__)
        replace_if(individual, pop, index_min)


##############################
# function steady_state
##############################
def steady_state(client, births, init_pop_size, pop_size,
                 representation,
                 problem, offspring_pipeline,
                 inserter=greedy_insert_into_pop,
                 count_nonviable=False,
                 evaluated_probe=None,
                 pop_probe=None,
                 context=context):
    """ Implements an asynchronous steady-state EA

    :param client: Dask client that should already be set-up
    :param births: how many births are we allowing?
    :param init_pop_size: size of initial population sent directly to workers
           at start
    :param pop_size: how large should the population be?
    :param representation: of the individuals
    :param problem: to be solved
    :param offspring_pipeline: for creating new offspring from the pop
    :param inserter: function with signature (new_individual, pop, popsize)
           used to insert newly evaluated individuals into the population;
           defaults to greedy_insert_into_pop()
    :param count_nonviable: True if we want to count non-viable individuals
           towards the birth budget
    :param evaluated_probe: is a function taking an individual that is given
           the next evaluated individual; can be used to print newly evaluated
           individuals
    :param pop_probe: is an optional function that writes a snapshot of the
           population to a CSV formatted stream ever N births
    :return: the population containing the final individuals
    """
    initial_population = representation.create_population(init_pop_size,
                                                          problem=problem)

    # fan out the entire initial population to dask workers
    as_completed_iter = eval_population(initial_population, client=client,
                                        context=context)

    # This is where we'll be putting evaluated individuals
    pop = []

    # Bookkeeping for tracking the number of births
    birth_counter = util.inc_births(context, start=len(initial_population))

    for i, evaluated_future in enumerate(as_completed_iter):

        evaluated = evaluated_future.result()

        if evaluated_probe is not None:
            # Give a chance to do something extra with the newly evaluated
            # individual, which is *usually* a call to
            # probe.log_worker_location, but can be any function that
            # accepts an individual as an argument
            evaluated_probe(evaluated)

        logger.debug('%d evaluated: %s %s', i, str(evaluated.genome),
                     str(evaluated.fitness))

        if not count_nonviable and not is_viable(evaluated):
            # If we don't want non-viable individuals to count towards the
            # birth budget, then we need to decrement the birth count that was
            # incremented when it was created for this individual since it
            # was broken in some way.
            birth_counter()

        inserter(evaluated, pop, pop_size)

        if pop_probe is not None:
            pop_probe(pop)

        if birth_counter.births() < births:
            # Only create offspring if we have the budget for one
            offspring = toolz.pipe(pop, *offspring_pipeline)

            logger.debug('created offspring: ')
            [logger.debug('%s', str(o.genome)) for o in offspring]

            # Now asynchronously submit to dask
            for child in offspring:
                future = client.submit(evaluate(context=context), child,
                                       pure=False)
                as_completed_iter.add(future)

            birth_counter(len(offspring))

    return pop
