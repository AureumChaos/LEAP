"""
    Tests for leap_ec.distrib.async for handling birth budgets
"""
import numpy as np
from distributed import Client

import leap_ec.ops as ops
from typing import Iterator
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.distrib.evaluate import evaluate, is_viable
from leap_ec.distrib.individual import DistributedIndividual
from leap_ec.distrib.asynchronous import steady_state
from leap_ec.global_vars import context
from leap_ec.individual import Individual
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.representation import Representation

def accumulate():
    """ This pipeline operator accumulates individuals as they move
        through the pipeline.

        TODO consider moving this to probe.py
    """
    inds = []

    def individuals():
        return inds

    def _accumulate(next_individual: Iterator):
        nonlocal inds

        while True:
            individual = next(next_individual)

            inds.append(individual)

            yield individual

    # convenient accessor for individuals
    _accumulate.individuals = individuals

    return _accumulate


representation = Representation(create_binary_sequence(3))

def test_meet_budget():
    """ This test is to ensure that we meet our birth budget exactly """
    with Client() as client:

        my_accumulate = accumulate()

        pop = steady_state(client=client,
                           births=4,
                           init_pop_size=2,
                           pop_size=2,
                           representation=representation,
                           problem=MaxOnes(),
                           offspring_pipeline=[
                               ops.random_selection,
                               ops.clone,
                               my_accumulate,
                               ops.pool(size=1)]
                           )

    inds = my_accumulate.individuals()

    assert len(pop) == 2


def test_meet_budget_count_nonviable():
    """ Birth budget counting non-viable individuals """
    pass

def test_meet_budget_do_not_count_nonviable():
    """ Birth budget without counting non-viable individuals """
    pass
