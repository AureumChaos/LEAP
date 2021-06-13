"""
    Tests for leap_ec.distrib.evaluate.
"""
from distributed import Client

from leap_ec.individual import Individual
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.distrib.evaluate import evaluate
from leap_ec.decoder import IdentityDecoder
from leap_ec.global_vars import context


def test_good_eval():
    """
        This is for testing a plain ole good individual to ensure that
        leap_ec.distrib.evaluate works for normal circumstances.
    """
    # set up a basic dask local cluster
    with Client() as client:
        # hand craft an individual that should evaluate fine
        # Let's try evaluating a single individual
        individual = Individual([1, 1], decoder=IdentityDecoder(),
                                     problem=MaxOnes())

        future = client.submit(evaluate(context=context),
                               individual)

        evaluated_individual = future.result()

        assert evaluated_individual.fitness == 2


def test_broken_individual_eval():
    """
        Test an individual that intentionally throws an exception during
        evaluation, which marks that individual has non-viable.

        TODO implement this
    """
    # set up a basic dask local cluster

    # hand craft an individual that should throw an exception on evaluation

    # evaluate that individual

    # check that the individual state is sane given it is non-viable

    pass
