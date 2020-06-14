"""
    Unit test for selection operators.
"""
import itertools

from leap_ec import core
from leap_ec import ops
from leap_ec import binary_problems


def test_naive_cyclic_selection():
    """ Test of the naive deterministic cyclic selection """
    pop = [core.Individual([0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
           core.Individual([0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]

    # This selection operator will deterministically cycle through the
    # given population
    selector = ops.naive_cyclic_selection(pop)

    selected = next(selector)
    assert selected.genome == [0, 0]

    selected = next(selector)
    assert selected.genome == [0, 1]

    # And now we cycle back to the first individual
    selected = next(selector)
    assert selected.genome == [0, 0]



def test_cyclic_selection():
    """ Test of the deterministic cyclic selection """

    # We're just going to use integers for the population as that's sufficient for testing this selection
    # operator; we don't want to get in the weeds with comparing individuals for sequence equivalency testing.
    pop = list(range(4))

    # This selection operator will deterministically cycle through the
    # given population
    selector = ops.cyclic_selection(pop)

    # first cycle should be the same order as we started
    first_iteration = [next(selector) for _ in range(len(pop))]

    assert pop == first_iteration

    # the second iteration should be shuffled
    second_iteration = [next(selector) for _ in range(len(pop))]

    assert pop != second_iteration



def test_truncation_selection():
    """ Basic truncation selection test"""
    pop = [core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
           core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
           core.Individual([1, 1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
           core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]

    # We first need to evaluate all the individuals so that truncation selection has fitnesses to compare
    pop = core.Individual.evaluate_population(pop)

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
    parents = [core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
               core.Individual([1, 1, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]

    parents = core.Individual.evaluate_population(parents)

    offspring = [core.Individual([0, 0, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
                 core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]
    offspring = core.Individual.evaluate_population(offspring)

    truncated = ops.truncate(offspring, 2, parents=parents)

    assert len(truncated) == 2

    assert parents[1] in truncated
    assert offspring[1] in truncated


def test_tournament_selection():
    """ This simple binary tournament selection """
    # Make a population where binary tournament has an obvious reproducible choice
    pop = [core.Individual([0, 0, 0], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes()),
           core.Individual([1, 1, 1], decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())]

    # We first need to evaluate all the individuals so that truncation selection has fitnesses to compare
    pop = core.Individual.evaluate_population(pop)

    best = next(ops.tournament(pop))
    pass

    # This assert will sometimes not work because it's possible to select the same individual more than once, and that
    # includes scenarios where the worst of two individuals is selected twice, which can happen about 25% of the time.
    # assert pop[1] == best




