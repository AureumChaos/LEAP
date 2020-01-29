#!/usr/bin/env python3
"""
    usage: simple.py [-h] [--verbose] [--workers WORKERS]
                 [--init-pop-size INIT_POP_SIZE] [--max-births MAX_BIRTHS]
                 [--pool-size POOL_SIZE] [--scheduler-file SCHEDULER_FILE]

Simple PEAL example of asynchronously distributing MAX ONES problem to workers

optional arguments:
  -h, --help            show this help message and exit
  --verbose, -v         Chatty output
  --workers WORKERS, -w WORKERS
                        How many workers?
  --init-pop-size INIT_POP_SIZE, -s INIT_POP_SIZE
                        What should the size of the initial population be?
                        Ideally this should be at least the same as the number
                        of workers to ensure that the worker pool is saturated
                        at the very start of the runs
  --max-births MAX_BIRTHS, -m MAX_BIRTHS
                        Maximum number of births before ending
  --pool-size POOL_SIZE, -p POOL_SIZE
                        The size of the evaluated individuals pool
  --scheduler-file SCHEDULER_FILE, -f SCHEDULER_FILE
                        The scheduler file used to coordinate between the
                        scheduler and workers. Specifying this option
                        automatically triggers non-local distribution of
                        workers, such as on a local cluster
"""
import logging
from pprint import pformat
import socket
import os
import argparse
import random
import uuid
from time import sleep

from dask.distributed import Client, LocalCluster

from leap import core
from leap import ops
from leap import binary_problems
from leap import util
import leap.parallel.parallel

# Create unique logger for this namespace
logger = logging.getLogger(__name__)

# default number of workers to evaluate individuals
DEFAULT_NUM_WORKERS = 5

# default number of initial population size; ideally should be the same as
# number of workers so that we saturate the worker pool right out of the gate.
DEFAULT_INIT_POP_SIZE = DEFAULT_NUM_WORKERS


class MyIndividual(core.Individual):
    """
        Just over-riding class to add some pretty printing stuff.

        TODO I should consider centrally defining this convenience class or
        its functionality
    """

    # True if we want to save what host and process was used to evaluation an
    # individual
    save_eval_environment = False

    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder, problem)

        # Used to uniquely identify this individual
        self.uuid = uuid.uuid4()

    def __repr__(self):
        return " ".join([str(self.uuid), str(self.birth), str(self.fitness),
                         "".join([str(x) for x in self.genome])])

    def is_viable(self):
        """ This is used by Parallel to ensure that we are considering "viable"
        individuals.

        That is, an individual may have been returned from a worker as *not*
        viable because its evaluation was interrupted by, say, doing a check-
        point. In which case, we do not want to insert it into the pool.

        TODO but ensure we have a mechanism in place to properly report and
        otherwise handle such individuals.  (And better define what we mean by
        "otherwise handle.")

        :return: True
        """
        return True

    def evaluate(self):
        """ Evaluate this individual, but with some additional logging thrown
        in.

        :return: evaluated individual
        """
        result = super().evaluate()

        # We sleep for a random number of seconds to test that we're actually
        # working asynchronously.
        sleep(random.randint(1, 6))

        logger.info('on %s in process %s evaluated %s', socket.gethostname(),
                    os.getpid(), str(self))

        if MyIndividual.save_eval_environment:
            with open(str(self.uuid) + '.csv', 'w') as save_file:
                save_file.write(
                    socket.gethostname() + ', ' + str(os.getpid()) + ', ' + str(
                        self.birth) + ', ' +
                    str(self.fitness) + ', ' + "".join(
                        [str(x) for x in self.encoding.decode()]) + '\n')

        return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple PEAL example of asynchronously distributing MAX '
                    'ONES problem to workers')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Chatty output')

    parser.add_argument('--workers', '-w', type=int,
                        default=DEFAULT_NUM_WORKERS, help='How many workers?')
    parser.add_argument('--init-pop-size', '-s', type=int,
                        default=DEFAULT_INIT_POP_SIZE,
                        help='What should the size of the initial population '
                             'be? Ideally this should be at least '
                             'the same as the number of workers to ensure '
                             'that the worker pool is saturated '
                             'at the very start of the runs')
    parser.add_argument('--max-births', '-m', type=int,
                        help='Maximum number of births before ending')
    parser.add_argument('--pool-size', '-p', type=int,
                        help='The size of the evaluated individuals pool')
    parser.add_argument('--scheduler-file', '-f',
                        help='The scheduler file used to coordinate between '
                             'the scheduler '
                             'and workers. Specifying this option '
                             'automatically triggers '
                             'non-local distribution of workers, such as on a '
                             'local '
                             'cluster')
    parser.add_argument('--save', action='store_true',
                        help='Save individuals to a log file named by their '
                             'UUID that saves '
                             'hostname and process ID during evaluation')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.save:
        MyIndividual.save_eval_environment = True

    logger.info(
        'workers: %s init pop size: %s max births: %s, pool size: %s, save: %s',
        args.workers, args.init_pop_size, args.max_births, args.pool_size,
        MyIndividual.save_eval_environment)

    my_max_ones = binary_problems.MaxOnes()
    my_decoder = core.IdentityDecoder()

    try:
        if args.scheduler_file:
            # We're wanting to submit workers onto other nodes, and *not* run
            # them locally because we went through the trouble of specifying
            # a scheduler file that the scheduler and workers will use to
            # coordinate with one another.
            logging.info('Using a remote distributed model')
            client = Client(scheduler_file=args.scheduler_file)
        else:
            logging.info('Using a local distributed model')
            cluster = LocalCluster(n_workers=args.workers, processes=False,
                                   silence_logs=logger.level)
            logger.info("Cluster: %s", cluster)
            client = Client(cluster)

        logger.info('Client: %s', client)

        my_parallel = leap.parallel.parallel.Parallel(client,
                                                      max_births=args.max_births,
                                                      pool_size=args.pool_size)

        final_pop = my_parallel.do(MyIndividual,
                                   initializer=core.create_binary_sequence(4),
                                   init_pop_size=args.init_pop_size,
                                   problem=my_max_ones, decoder=my_decoder)

        logger.info('Final pop: %s', pformat(final_pop))
    except Exception as e:
        logger.critical(str(e))
    finally:
        client.close()

    logger.info('Done.')
