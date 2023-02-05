#!/usr/bin/env python3
""" Simple example of using leap_ec.distrib.asynchronous.steady_state()

    usage: simple_async_distributed.py [-h] [--verbose]
                                   [--track-workers-file TRACK_WORKERS_FILE]
                                   [--track-pop-file TRACK_POP_FILE]
                                   [--update-interval UPDATE_INTERVAL]
                                   [--workers WORKERS]
                                   [--init-pop-size INIT_POP_SIZE]
                                   [--max-births MAX_BIRTHS]
                                   [--pop-size POP_SIZE]
                                   [--scheduler-file SCHEDULER_FILE]
                                   [--length LENGTH]

Simple example of asynchronously distributing MAX ONES problem to workers

optional arguments:
  -h, --help            show this help message and exit
  --verbose, -v         Chatty output
  --track-workers-file TRACK_WORKERS_FILE, -t TRACK_WORKERS_FILE
                        Optional file to write CSV of what host and process ID
                        was associated with each evaluation
  --track-pop-file TRACK_POP_FILE
                        Optional CSV file to take regular interval snapshots
                        of the population ever --update-intervals
  --update-interval UPDATE_INTERVAL
                        If using --track-pop-file, how many births before
                        writing an update to the specified file
  --workers WORKERS, -w WORKERS
                        How many workers?
  --init-pop-size INIT_POP_SIZE, -s INIT_POP_SIZE
                        What should the size of the initial population be?
                        Ideally this should be at least the same as the number
                        of workers to ensure that the worker pool is saturated
                        at the very start of the runs
  --max-births MAX_BIRTHS, -m MAX_BIRTHS
                        Maximum number of births before ending
  --pop-size POP_SIZE, -b POP_SIZE
                        The size of the evaluated individuals pop
  --scheduler-file SCHEDULER_FILE, -f SCHEDULER_FILE
                        The scheduler file used to coordinate between the
                        scheduler and workers. Specifying this option
                        automatically triggers non-local distribution of
                        workers, such as on a local cluster
  --length LENGTH, -l LENGTH
                        Genome length
"""
import argparse
import logging
import os
from pprint import pformat

from distributed import Client, LocalCluster

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.distrib import DistributedIndividual
from leap_ec.distrib import asynchronous
from leap_ec.distrib.logger import WorkerLoggerPlugin
from leap_ec.distrib.probe import log_worker_location, log_pop


# Create unique logger for this namespace
logger = logging.getLogger(__name__)

# default number of workers to evaluate individuals
DEFAULT_NUM_WORKERS = 5

# default number of initial population size; ideally should be the same as
# number of workers so that we saturate the worker pool right out of the gate.
DEFAULT_INIT_POP_SIZE = DEFAULT_NUM_WORKERS

# Default number of births to update --track-pop-file
DEFAULT_UPDATE_INTERVAL = 5

# When running the test harness, just run for two steps
# (we use this to quickly ensure our examples don't get bitrot)
if os.environ.get(test_env_var, False) == 'True':
    DEFAULT_MAX_BIRTHS = 2
else:
    DEFAULT_MAX_BIRTHS = 100

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simple example of asynchronously distributing MAX '
                    'ONES problem to workers')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Chatty output')
    parser.add_argument('--track-workers-file', '-t',
                        help='Optional file to write CSV of what host and '
                        'process ID was associated with each evaluation')
    parser.add_argument('--track-pop-file',
                        help='Optional CSV file to take regular interval'
                        ' snapshots of the population ever --update-intervals')
    parser.add_argument('--update-interval', type=int,
                        default=DEFAULT_UPDATE_INTERVAL,
                        help='If using --track-pop-file, how many births before'
                             ' writing an update to the specified file')
    parser.add_argument('--workers', '-w', type=int,
                        default=DEFAULT_NUM_WORKERS, help='How many workers?')
    parser.add_argument('--init-pop-size', '-s', type=int,
                        default=DEFAULT_INIT_POP_SIZE,
                        help='What should the size of the initial population '
                             'be? Ideally this should be at least '
                             'the same as the number of workers to ensure '
                             'that the worker pool is saturated '
                             'at the very start of the runs')
    parser.add_argument('--max-births', '-m', type=int, default=DEFAULT_MAX_BIRTHS,
                        help='Maximum number of births before ending')
    parser.add_argument('--pop-size', '-b', type=int, default=5,
                        help='The size of the evaluated individuals pop')
    parser.add_argument('--scheduler-file', '-f',
                        help='The scheduler file used to coordinate between '
                             'the scheduler '
                             'and workers. Specifying this option '
                             'automatically triggers '
                             'non-local distribution of workers, such as on a '
                             'local '
                             'cluster')
    parser.add_argument('--length', '-l', type=int, default=5,
                        help='Genome length')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

    logger.info(
        'workers: %s init pop size: %s max births: %s, pop size: %s',
        args.workers, args.init_pop_size, args.max_births, args.pop_size)

    track_workers_func = track_pop_func = None
    client = None  # So the `finally` block below works if client never gets declared

    try:
        if args.scheduler_file:
            # We're wanting to submit workers onto other nodes, and *not* run
            # them locally because we went through the trouble of specifying
            # a scheduler file that the scheduler and workers will use to
            # coordinate with one another.
            logger.info('Using a remote distrib model')
            client = Client(scheduler_file=args.scheduler_file)
        else:
            logger.info('Using a local distrib model')
            cluster = LocalCluster(n_workers=args.workers, processes=False,
                                   silence_logs=logger.level)
            logger.info("Cluster: %s", cluster)
            client = Client(cluster)

        client.register_worker_plugin(WorkerLoggerPlugin(verbose=args.verbose))

        logger.info('Client: %s', client)

        if args.track_workers_file:
            track_workers_stream = open(args.track_workers_file, 'w')
            track_workers_func = log_worker_location(track_workers_stream)

        if args.track_pop_file is not None:
            track_pop_stream = open(args.track_pop_file, 'w')
            track_pop_func = log_pop(args.update_interval, track_pop_stream)

        final_pop = asynchronous.steady_state(client,
                                              max_births=args.max_births,
                                              init_pop_size=args.init_pop_size,
                                              pop_size=args.pop_size,

                                              representation=Representation(
                                                  initialize=create_binary_sequence(
                                                      args.length),
                                                  individual_cls=DistributedIndividual),

                                              problem=MaxOnes(),

                                              offspring_pipeline=[
                                                  ops.random_selection,
                                                  ops.clone,
                                                  mutate_bitflip(expected_num_mutations=1),
                                                  ops.pool(size=1)],

                                              evaluated_probe=track_workers_func,
                                              pop_probe=track_pop_func)

        logger.info('Final pop: \n%s', pformat(final_pop))
    except Exception as e:
        logger.critical(str(e))
        raise e
    finally:
        if client is not None:
            # Because an exception could have been thrown such that client does
            # not exist.
            client.close()

        if track_workers_func is not None:
            track_workers_stream.close()
        if track_pop_func is not None:
            track_pop_stream.close()

    logger.info('Done.')
