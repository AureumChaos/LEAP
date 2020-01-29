#!/usr/bin/env python3
"""
    LEAP sub-module supporting parallel evaluations using dask.

"""
import random
import logging
from pprint import pformat

import toolz

from queue import PriorityQueue

from dask.distributed import as_completed

from leap import ops
from leap import util

# Create unique logger for this namespace
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# maximum number of individuals in the pool at once.
DEFAULT_POOL_SIZE = 5

# how many children beyond initial population do we want to spawn?
DEFAULT_MAX_BIRTHS = 100


class Parallel:
    """ Allows for concurrent, distributed fitness evaluations using a
    steady-state approach.
    """

    def __init__(self, client, max_births, pool_size):
        """

        :param client: an open dask client for distributing evals
        :param max_births: most number of evals
        :param pool_size: size of the pool of evaluated individuals
        """
        self.client = client
        self.max_births = max_births

        # This will be decremented in do() and is used by halt() to determine
        # when we're done.
        self.births_remaining = self.max_births

        self.max_pool_size = pool_size

        # current "pool" of individuals that always contains the fittest
        # individuals so far
        self.current_population = PriorityQueue(self.max_pool_size)

        # We want to tag each individual with a unique birth ID
        self.birth_brander = util.birth_brander()

    @classmethod
    def evaluate(cls, individual):
        """ This is the worker task for evaluating the given individual

        TODO not a fan of returning tuple. Review the need for this.

        Also, need to consider returning a lot more than just the fitness and
        individual, but maybe other ancillary information.

        :param individual: to be evaluated
        :return: individual.fitness, individual
        """
        individual.evaluate()

        return individual.fitness, individual

    def halt(self):
        """ Optional over-rideable function to allow for tailored stopping
        criteria.

        E.g., when it's time to checkpoint before the birth budget has been met.

        self.births_remaining is decremented in do() for each viable created
        individual.

        :return: True if stopping criteria met, else False
        """
        # Though I *should* == 0, I'm paranoid that somehow this could fall
        # negative due to race conditions; which shouldn't happen, but we all
        # know how situations arise that shouldn't happen.
        return self.births_remaining < 1

    def create_child(self, population):
        """ Creates a child from the given population

        You will want to over-ride this in your own implementation.

        The pattern for this implementation is that the expected output
        should be a *single individual*, hence the pool(size=1) should be the
        last pipeline operator.

        TODO We can probably do a better job of making the pipeline to better
        impose pre-requisites.

        :param population: the current pool of already evaluated individuals
        :return: newly created child
        """
        return toolz.pipe(population,
                          ops.random_selection,
                          ops.clone,
                          self.birth_brander,
                          ops.mutate_bitflip,
                          ops.pool(size=1))


    def check_if_evaluated(self, individual):
        """ Have an opportunity to handle individuals that have just been
        evaluated, and just before they're considered to be added to the pool.

        This is to be optionally over-ridden, and typically used to record
        these individuals to a file as they're evaluated.

        The ability to return True or False can be over-ridden by a subclass,
        and may be useful in such scenarios as supporting checkpoints and
        restarts.  That is, the given individual may not be done evaluating
        because the evaluation was interrupted by a checkpoint

        :param individual: to be handled

        :return: True if the individual is suitable to be added to the pool,
        else False
        """
        return individual.fitness is not None

    def create_initial_population(self, class_individual, init_pop_size,
                                  initializer,
                                  problem, decoder):
        """ This can be over-ridden for, say, creating a population from a
        checkpoint.

        :param class_individual: is Class of Individual to be replicated
        :param init_pop_size: how large do we want the initial population to be?
        :param initializer: mechanism for creating new individuals
        :param problem: we are trying to solve
        :param decoder: for decoding genome to meaningful values
        :return: The initial population to be sent in bulk to the workers for
        evaluation
        """
        initial_population = class_individual.create_population(init_pop_size,
                                                  initialize=initializer,
                                                  problem=problem,
                                                  decoder=decoder)
        # Just run the initial population through the birth ID brander to
        # ensure we start out with proper birth IDS for each of them.
        branded_population = self.birth_brander.brand_population(initial_population)

        return branded_population

    def do(self, class_individual, initializer, init_pop_size, problem, decoder):
        """ Update a pool of individuals from dask workers until a budget of
        births is met, or some other stopping criteria.

        TODO Add in optional other stopping criteria. Started with self.halt(
        ), but need to incorporate it into logic.

        :param class_individual: the class from which we will be creating
        individuals
        :param initializer: to create individuals
        :param init_pop_size: is the size of the initial population
        :param problem: that we're trying to solve
        :param decoder: how the individuals represent posed solutions for
        the problem

        :return: final population
        """
        logger.debug('do() init_pop_size %d', init_pop_size)

        # create initial population
        parent_population = self.create_initial_population(class_individual,
                                                           init_pop_size,
                                                           initializer,
                                                           problem, decoder)

        logger.debug('parents: %s', pformat(parent_population))

        # farm out initial population to worker nodes for evaluation
        worker_futures = self.client.map(Parallel.evaluate, parent_population)

        iterator = as_completed(worker_futures)

        # grind through all the evaluated individuals until we've achieved
        # MAX_REPRODUCTIONS spawned offspring
        for res in iterator:

            x = res.result()

            # Subclasses may want to do something special with this newly
            # evaluated individual just before it's considered to be added to
            # the pool.  finished_evaluating will be false if, say,
            # the individual has been checkpointed and has not yet been fully
            # evaluated, so there's no point in adding it to the pool.
            finished_evaluating = self.check_if_evaluated(x[1])

            logger.debug('got result: %s %s', str(x[0]), str(x[1]))

            if finished_evaluating:
                # Only consider adding fully evaluated individuals to the
                # pool of evaluated individuals

                if self.current_population.full():
                    # remove the least fit individual; unless it's weaker; if
                    # equal, then coin toss whether to replace them then
                    # insert the new guy
                    logger.info('Pool full ... voting someone off the island')
                    weakest = self.current_population.get()
                    logger.debug('current weakest: %s', str(weakest))
                    if weakest[0] < x[0]:
                        logger.debug('%s < %s so inserting: %s',
                                     str(weakest[1]), str(x[1]), str(x))
                        self.current_population.put(x)
                    elif weakest[0] == x[0]:
                        logger.debug('%s == %s so flipping coin',
                                     str(weakest[1]), str(x[1]))
                        if random.random() < 0.5:
                            self.current_population.put(x)
                            logger.debug('%s won the toss', str(x[1]))
                        else:
                            self.current_population.put(weakest)
                            logger.debug('%s won the toss', str(weakest[1]))
                    else:
                        logger.debug('%s >= %s so re-inserting weakest',
                                     str(weakest[1]), str(x[1]))
                        self.current_population.put(weakest)
                else:
                    # just insert the individual
                    logger.debug('Inserting: %s', str(x))
                    self.current_population.put(x)

                logger.info('>>> current population: \n%s\n',
                            pformat(self.current_population.queue))

                # Only create offspring if we have budget left for that,
                # however, we also don't want to count non-viable individuals
                # towards our birth budget.  x[1] because the second element
                # of the returned tuple is the actual individual; x[0] is its
                # fitness broken out to make it easier for queue management.
                if x[1].is_viable():
                    self.births_remaining -= 1

            if not self.halt():

                # pull out just the individuals and not their fitness
                scratch_pop = list(
                    toolz.pluck(1, self.current_population.queue))

                if len(scratch_pop) == 1:
                    logger.info('spawning first child from pool')

                child = self.create_child(scratch_pop)

                logger.info('new birth: %s', str(child))

                # child[0] because the pipeline essentially returns a
                # population of one, and so we just unpack that one kid and
                # pass it along to get evaluated
                new_evaluation = self.client.submit(Parallel.evaluate, child[0])
                iterator.add(new_evaluation)
            else:
                # We're draining down the workers, likely due to exhausting
                # birth budget, so let the pending evaluations finish.
                pass

        logger.info('final population: \n%s\n',
                    pformat(self.current_population.queue))

        # extract the individuals from inside the Queue and return that
        return list(toolz.pluck(1, self.current_population.queue))


if __name__ == '__main__':
    pass
