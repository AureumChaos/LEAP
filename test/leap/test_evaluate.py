"""
    Unit tests for cloning
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from leap import core
from leap import binary_problems
from leap import ops


def test_simple_evaluate():
    # Let's try evaluating a single individual
    pop = []

    pop.append(core.Individual([1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    evaluated_individual, args, kwargs = next(ops.evaluate(iter(pop)))

    assert evaluated_individual.fitness == 2
    assert args == ()
    assert kwargs == {}



def test_evaluate_with_args():
    pop = []
    pop.append(core.Individual([1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    evaluated_individual, args, kwargs = next(ops.evaluate(iter(pop), 99, foo='bar'))

    assert evaluated_individual.fitness == 2
    assert args == (99,)
    assert kwargs['foo'] == 'bar'


def test_evaluate():

    # We need an Individual with a simple encoding and a corresponding
    # problem so that we have something with which to evaluate.

    pop = [core.Individual([1, 1], decoder=core.IdentityDecoder(),
                           problem=binary_problems.MaxOnes())]

    # The one individual hasn't been evaluated yet, so its fitness
    # should be None

    assert pop[0].fitness is None

    # Since we're using generators, let's create a new sequence with
    # the hopefully now evaluated individual.  Note that evaluate() returns
    # the evaluated individual *and* the optional args and kwargs.  Since
    # we're doing a test, we strip out the args and kwargs to get at just
    # the individual.
    # TODO add a test to ensure that the args and kwargs get properly
    # propagated.

    new_pop = []

    i = iter(pop) # this must be declared *outside* the for loop, else there will be an extra iteration
    for individual, args, kwargs in ops.evaluate(i):
        new_pop.append(individual)

    # Which should now have a fitness.
    assert new_pop[0].fitness == 2

    # I've been having problems with extra iterations
    assert len(new_pop) == len(pop)

    # And so to show that there's no copying, we can similarly refer to
    # the same individual in the original sequence to show that, yes,
    # it really did get evaluated.

    assert pop[0].fitness == 2


