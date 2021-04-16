#!/usr/bin/env python3
"""
    Subclass of core.Individual that adds some state relevant for distrib
    runs.

    Adds:

    * uuid for each individual
    * birth ID, a unique birth number; first individual has ID 0, the last N-1.
"""
import itertools
import uuid

from leap_ec.individual import RobustIndividual


class DistributedIndividual(RobustIndividual):
    # Tracks unique birth ID for each newly created individual
    birth_id = itertools.count()

    """
        Core individual that has unique UUID and birth ID.
    """

    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder, problem)

        self.uuid = uuid.uuid4()

        self.birth_id = next(DistributedIndividual.birth_id)

        # These are set in evaluate.evaluate(), so these are just to inform
        # you that that function will set these variables.
        self.start_eval_time = None
        self.stop_eval_time = None
        self.is_viable = False
        self.exception = None

    def __str__(self):
        return f'{self.uuid} birth: {self.birth_id} fitness: {self.fitness!s} ' \
               f'genome: {self.genome!s} '
