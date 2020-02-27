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
import argparse

from dask.distributed import Client, LocalCluster

from leap import core
from leap import ops
from leap import binary_problems
from leap.distributed import asynchronous
from leap.distributed.logging import WorkerLoggerPlugin

# Create unique logger for this namespace
logger = logging.getLogger(__name__)

# default number of workers to evaluate individuals
DEFAULT_NUM_WORKERS = 5

# default number of initial population size; ideally should be the same as
# number of workers so that we saturate the worker pool right out of the gate.
DEFAULT_INIT_POP_SIZE = DEFAULT_NUM_WORKERS


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple example of asynchronously distributing MAX '
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
    parser.add_argument('--bag-size', '-b', type=int,
                        help='The size of the evaluated individuals bag')
    parser.add_argument('--scheduler-file', '-f',
                        help='The scheduler file used to coordinate between '
                             'the scheduler '
                             'and workers. Specifying this option '
                             'automatically triggers '
                             'non-local distribution of workers, such as on a '
                             'local '
                             'cluster')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(
        'workers: %s init pop size: %s max births: %s, bag size: %s',
        args.workers, args.init_pop_size, args.max_births, args.bag_size)

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

        client.register_worker_plugin(WorkerLoggerPlugin(verbose=args.verbose))

        logger.info('Client: %s', client)

        final_pop = asynchronous.steady_state(client, births=9, init_pop_size=5,
                                              bag_size=3,
                                              initializer=core.create_binary_sequence(
                                                  4),
                                              decoder=core.IdentityDecoder(),
                                              problem=binary_problems.MaxOnes(),
                                              offspring_pipeline=[
                                                  ops.random_selection,
                                                  ops.clone,
                                                  ops.mutate_bitflip,
                                                  ops.pool(size=1)])

        logger.info('Final pop: \n%s', pformat(final_pop))
    except Exception as e:
        logger.critical(str(e))
    finally:
        client.close()

    logger.info('Done.')
