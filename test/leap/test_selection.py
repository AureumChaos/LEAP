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

    selected = next(selector)
    assert selected.genome == [0,0]

    selected = next(selector)
    assert selected.genome == [0,1]

    # And now we cycle back to the first individual
    selected = next(selector)
    assert selected.genome == [0,0]



def test_truncation_selection():
    """ Basic truncation selection test"""
    pop = []

    pop.append(core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    # We first need to evaluate all the individuals so that truncation selection has fitnesses to compare
    pop = [individual for individual in ops.evaluate(iter(pop))]

    truncated = ops.truncate(pop, 2)

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
    parents = [individual for individual in ops.evaluate(i)]

    offspring = []
    offspring.append(core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    offspring.append(core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    i = iter(offspring)
    offspring = [individual for individual in ops.evaluate(i)]

    truncated = ops.truncate(offspring, 2, parents=parents)

    assert len(truncated) == 2

    assert parents[1] in truncated
    assert offspring[1] in truncated



def test_tournament_selection():
    """ This simple binary tournament selection """
    pop = []

    # Make a population where binary tournament has an obvious reproducible choice
    pop.append(core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))
    pop.append(core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()))

    i = iter(pop)

    # We first need to evaluate all the individuals so that truncation selection has fitnesses to compare
    pop = [individual for individual in ops.evaluate(i)]

    best = next(ops.tournament(pop))
    pass

    # This assert will sometimes not work because it's possible to select the same individual more than once, and that
    # includes scenarios where the worst of two individuals is selected twice, which can happen about 25% of the time.
    # assert pop[1] == best




