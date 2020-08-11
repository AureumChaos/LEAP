#!/usr/bin/env python3
"""
    This provides a dask logging plugin that reports the hostname, worker ID,
    and process ID for each worker.  This is useful for checking that all
    workers have been sanely assigned to targeted resources.

    Note that once this plugin is installed that dask will ensure that each
    worker restarted after a failure gets the plugin re-installed, too.
"""
import os
import platform
import logging

import dask
from dask.distributed import Client, as_completed, LocalCluster, get_worker, WorkerPlugin


class EvaluatorLogFilter(logging.Filter):
    """ Convenience for adding hostname and worker ID to log messages

    Cribbed from https://stackoverflow.com/questions/55584115/python-logging-how-to-track-hostname-in-logs
    """

    def __init__(self):
        super().__init__()

        self.hostname = platform.node()
        self.process_id = os.getpid()

    def filter(self, record):
        record.hostname = self.hostname
        record.process_id = self.process_id

        return True


# We want the *same* logger used for all workers, so declare it here, and then
# ensure that all the workers point to this one.
logger = logging.getLogger(__name__)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.addFilter(EvaluatorLogFilter())

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(hostname)s - %(process_id)s - %(levelname)s : %(message)s',
    style='%')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


class WorkerLoggerPlugin(WorkerPlugin):
    """
        This dask worker plugin adds a logger for each worker that reports
        the hostname, worker ID, and process ID.

        Usage:

        client.register_worker_plugin(WorkerLoggerPlugin()) after dask client
        is setup.

        Then in code sent to worker:

        worker = get_worker()
        worker.logger.info('This is a log message')
    """

    def __init__(self, verbose=False, *args, **kwargs):
        """
        :param verbose: is True if you want DEBUG level output
        :param args: n/a
        :param kwargs: n/a
        """
        super().__init__()
        self.verbose = verbose

    def setup_logger(self, worker):
        # We need to create and attach a logger _to each worker_ since
        # they'll be living in their own process/thread space.
        worker.logger = logger

        if self.verbose:
            worker.logger.setLevel(logging.DEBUG)
        else:
            worker.logger.setLevel(logging.INFO)

    def setup(self, worker: dask.distributed.Worker):
        """ This is invoked once for each worker on their startup. The
            scheduler will also ensure that all workers invoke this.
         """
        if hasattr(worker, 'logger'):
            worker.logger.warning('already has a logger')
        else:
            # Create an attach a logger that will echo the hostname and
            # unique dask worker id with each log message along with a
            # timestamp
            self.setup_logger(worker)
            worker.logger.info(f'worker setup for {worker.id}')

    def teardown(self, worker: dask.distributed.Worker):
        worker.logger.info(f'Tearing down worker {worker.id}')
