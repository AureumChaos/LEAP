"""
    Tests for leap_ec.distrib.async for handling birth budgets
"""
import logging
logging.basicConfig(level=logging.DEBUG)

from distributed import Client, LocalCluster

import leap_ec.ops as ops
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.distrib.asynchronous import steady_state
from leap_ec.distrib.individual import DistributedIndividual
from leap_ec.problem import ScalarProblem
from leap_ec.representation import Representation


def accumulate():
    """ This pipeline operator accumulates individuals as they move
        through the pipeline.

        TODO consider moving this to probe.py
    """
    inds = []

    def individuals():
        return inds

    def _accumulate(individual):
        nonlocal inds
        inds.append(individual)

    # convenient accessor for individuals
    _accumulate.individuals = individuals

    return _accumulate


representation = Representation(create_binary_sequence(3),
                                individual_cls=DistributedIndividual)

def test_meet_budget():
    """ This test is to ensure that we meet our birth budget exactly """
    with Client() as client:
        # We accumulate all the evaluated individuals the number of
        # which should be equal to our birth budget of four.
        my_accumulate = accumulate()

        pop = steady_state(client=client,
                           max_births=4,
                           init_pop_size=2,
                           pop_size=2,
                           representation=representation,
                           problem=MaxOnes(),
                           evaluated_probe=my_accumulate,
                           offspring_pipeline=[
                               ops.random_selection,
                               ops.clone,
                               ops.pool(size=1)]
                           )

    inds = my_accumulate.individuals()

    assert len(inds) == 6


class Counter:
    """ From https://distributed.dask.org/en/stable/actors.html

        Used to manage birth counts in a concurrent safe setting.
     """
    n = 0

    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1
        return self.n

    def add(self, x):
        self.n += x
        return self.n


class BrokenProblem(ScalarProblem):
    """ Intentionally throw an exception at an interval

        The intent is to deterministically create non-viable
        individuals caused by throwing an exception during
        evaluations at a predetermined interval.
    """
    def __init__(self, n, counter_agent):
        """ :param n: throw an exception at the `n`th eval """
        super().__init__(True)
        self.n = n
        self.counter_agent = counter_agent
    def evaluate(self, phenome, *args, **kwargs):
        # Should lock the Actor to prevent race conditions on getting
        # a unique count.
        # counter_future = self.counter_agent.result()
        counter_future = self.counter_agent.increment()
        count = counter_future.result()
        if count == self.n:
            # Still do the increment before signaling this is a
            # "bad" eval
            raise RuntimeError('Dummy Exception')
        return count


def test_meet_budget_count_nonviable():
    """ Birth budget counting non-viable individuals """
    with LocalCluster(processes=False, threads_per_worker=1, n_workers=1) as cluster:
        with Client(cluster) as client:
            # Create a Counter on a worker
            future = client.submit(Counter, actor=True)
            counter = future.result()  # Get back a pointer to that object

            # We accumulate all the evaluated individuals the number of
            # which should be equal to our birth budget of four.
            my_accumulate = accumulate()

            pop = steady_state(client=client,
                            max_births=4,
                            init_pop_size=2,
                            pop_size=2,
                            representation=representation,
                            problem=BrokenProblem(1, counter),
                            evaluated_probe=my_accumulate,
                            count_nonviable=False,
                            offspring_pipeline=[
                                ops.random_selection,
                                ops.clone,
                                ops.pool(size=1)]
                            )

    inds = my_accumulate.individuals()

    assert len(inds) == 7

def test_meet_budget_do_not_count_nonviable():
    """ Birth budget without counting non-viable individuals """
    with Client(direct_to_workers=True) as client:
        # Create a Counter on a worker
        future = client.submit(Counter, actor=True)
        counter = future.result()  # Get back a pointer to that object

        # We accumulate all the evaluated individuals the number of
        # which should be equal to our birth budget of four.
        my_accumulate = accumulate()

        pop = steady_state(client=client,
                           max_births=4,
                           init_pop_size=2,
                           pop_size=2,
                           representation=representation,
                           problem=BrokenProblem(3, counter),
                           evaluated_probe=my_accumulate,
                           count_nonviable=True, # NOTE THIS CHANGE
                           offspring_pipeline=[
                               ops.random_selection,
                               ops.clone,
                               ops.pool(size=1)]
                           )

    inds = my_accumulate.individuals()

    assert len(inds) == 6
