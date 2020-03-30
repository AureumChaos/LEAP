#!/usr/bin/env python3
"""
    A collection of probe functions tailored for distributed evaluation
"""
import csv
import sys


def log_worker_location(stream=sys.stdout, header=True):
    """
    When debugging dask distribution configurations, this function can be used
    to track what machine and process was used to evaluate a given
    individual

    Suitable for being passed as the `evaluated_probe` argument for
    leap.distributed.asynchronous.steady_state().

    :param stream: to which we want to write the machine details
    :param header: True if we want a header for the CSV file
    :return: a function for recording where individuals are evaluated
    """
    writer = csv.DictWriter(stream, fieldnames=['hostname', 'pid', 'uuid', 'birth_id', 'fitness'])

    if header:
        writer.writeheader()

    def write_record(individual):
        """ This writes a row to the CSV for the given individual

        evaluate() will tack on the hostname and pid for the individual.  The
        uuid should also be part of the distributed.Individual, too.

        :param individual: to be written to stream
        :return: None
        """
        nonlocal writer

        writer.writerow({'hostname' : individual.hostname,
                         'pid' : individual.pid,
                         'uuid' : individual.uuid,
                         'birth_id' : individual.birth_id,
                         'fitness' : individual.fitness})

    return write_record
