"""
    Unit test for selection operators.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from leap import core
from leap import ops
from leap import binary_problems


def test_naive_cyclic_selection():
    """ Test of the naive deterministic cyclic selection """
    pop = []

    pop.append(core.Individual([0, 0]))
    pop.append(core.Individual([0, 1]))

    # This selection operator will deterministically cycle through the
    # given population
    selector = ops.naive_cyclic_selection_generator(pop)

    selected, args, kwargs = next(selector)
    assert selected.genome == [0,0]

    selected, args, kwargs = next(selector)
    assert selected.genome == [0,1]

    # And now we cycle back to the first individual
    selected, args, kwargs = next(selector)
    assert selected.genome == [0,0]



def test_truncation_selection():
    """ Basic truncation selection test"""
    pop = []

    pop.append(core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    i = iter(pop)

    # We first need to evaluate all the individuals so that truncation selection has fitnesses to compare
    pop = [individual for individual, args, kwargs in ops.evaluate(i)]

    i = iter(pop)
    truncated = ops.truncate(i, 2)

    assert len(truncated) == 2

    # Just to make sure, check that the two best individuals from the original population are in the selected population
    assert pop[2] in truncated
    assert pop[3] in truncated


def test_truncation_parents_selection():
    """ Test (mu + lambda), i.e., parents competing with offspring

    Create parent and offspring populations such that each has an "best" individual that will be selected by
    truncation selection.
    """
    parents = []

    parents.append(core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    parents.append(core.Individual([1, 1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    i = iter(parents)
    parents = [individual for individual, args, kwargs in ops.evaluate(i)]

    offspring = []
    offspring.append(core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    offspring.append(core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    i = iter(offspring)
    offspring = [individual for individual, args, kwargs in ops.evaluate(i)]

    parents_iter = iter(parents)
    offspring_iter = iter(offspring)

    truncated = ops.truncate(offspring_iter, 2, parents=parents_iter)

    assert len(truncated) == 2

    assert parents[1] in truncated
    assert offspring[1] in truncated


def test_truncation_selection_pipeline_args():
    """ test capability of passing along pipeline arguments for truncation selection
    """

    pop = []

    pop.append(core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    i = iter(pop)

    # We first need to evaluate all the individuals so that truncation selection has fitnesses to compare
    pop = [individual for individual, args, kwargs in ops.evaluate(i)]

    i = iter(pop)
    truncated, args, kwargs = ops.truncate(i, 2, 42, foo='bar')

    assert len(truncated) == 2

    # Just to make sure, check that the two best individuals from the original population are in the selected population
    assert pop[2] in truncated
    assert pop[3] in truncated

    assert args == (42,)
    assert kwargs['foo'] == 'bar'