"""
    Unit test for selection operators.
"""
import random
from math import nan

import numpy as np
import pytest

from leap_ec import Individual
from leap_ec import ops, statistical_helpers
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.data import test_population
from leap_ec.real_rep.problems import SpheroidProblem


##############################
# Tests for sus_selection()
##############################
def test_sus_selection1():
    ''' Test of a deterministic case of stochastic universal sampling '''
    # Make a population where sus_selection has an obvious
    # reproducible choice
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    pop = Individual.evaluate_population(pop)
    # This selection operator will always choose the [1, 1, 1] individual
    # since [0, 0, 0] has zero fitness
    selector = ops.sus_selection(pop)

    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])

    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])

    # run one more time to test shuffle
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])


@pytest.mark.stochastic
def test_sus_selection_shuffle():
    ''' Test of a stochastic case of SUS selection '''
    # Make a population where sus_selection has an obvious
    # reproducible choice
    # Proportions here should be 1/4 and 3/4, respectively
    pop = [Individual([0, 1, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    # Assign a unique identifier to each individual
    pop[0].id = 0
    pop[1].id = 1

    # We first need to evaluate all the individuals so that
    # selection has fitnesses to compare
    pop = Individual.evaluate_population(pop)
    selected = ops.sus_selection(pop)

    N = 1000
    p_thresh = 0.1
    observed_dist = statistical_helpers.collect_distribution(
        lambda: next(selected).id, samples=N)
    expected_dist = {pop[0].id: 0.25*N, pop[1].id: 0.75*N}
    print(f"Observed: {observed_dist}")
    print(f"Expected: {expected_dist}")
    assert(statistical_helpers.stochastic_equals(expected_dist,
                                                 observed_dist, p=p_thresh))


def test_sus_selection_offset():
    ''' Test of SUS selection with a non-default offset '''
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    # evaluate population and negate fitness of second individual
    pop = Individual.evaluate_population(pop)
    pop[1].fitness = -pop[1].fitness

    # now we try to evaluate normally (this should throw a ValueError)
    # due to the negative fitness
    with pytest.raises(ValueError):
        selector = ops.sus_selection(pop)
        selected = next(selector)
    # it should work by setting the offset to +3
    # this adds 3 to each fitness value, making the second
    # individual's fitness 0.
    selector = ops.sus_selection(pop, offset=3)

    # we expect the first individual to always be selected
    # since the new zero point is now -3.
    selected = next(selector)
    assert np.all(selected.genome == [0, 0, 0])

    selected = next(selector)
    assert np.all(selected.genome == [0, 0, 0])


def test_sus_selection_pop_min():
    ''' Test of SUS selection with pop-min offset '''
    # Create a population of positive fitness individuals
    # scaling the fitness by the population minimum makes it so the
    # least fit member never gets selected.
    pop = [Individual([0, 1, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    pop = Individual.evaluate_population(pop)

    selector = ops.sus_selection(pop, offset='pop-min')

    # we expect that the second individual is always selected
    # since the new zero point will be at the minimum fitness
    # of the population
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])

    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])


def test_sus_selection_custom_key():
    ''' Test of SUS selection with custom evaluation '''
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    def custom_key(individual):
        ''' Returns fitness based on MaxZeros '''
        return np.count_nonzero(individual.genome == 0)

    pop = Individual.evaluate_population(pop)
    selector = ops.sus_selection(pop, key=custom_key)

    # we expect the first individual to always be selected
    # since its genome is all 0s
    selected = next(selector)
    assert np.all(selected.genome == [0, 0, 0])

    selected = next(selector)
    assert np.all(selected.genome == [0, 0, 0])


def test_sus_selection_num_points():
    ''' Test of SUS selection with varying `n` random points '''
    # the second individual should always be selected
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    pop = Individual.evaluate_population(pop)
    # with negative points
    with pytest.raises(ValueError):
        selector = ops.sus_selection(pop, n=-1)
        selected = next(selector)

    # with n = None (default)
    selector = ops.sus_selection(pop, n=None)
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])

    # with n less than len(population)
    selector = ops.sus_selection(pop, n=1)
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])

    # with n greater than len(population)
    selector = ops.sus_selection(pop, n=3)
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])


##############################
# Tests for proportional_selection()
##############################
def test_proportional_selection1():
    ''' Test of a deterministic case of proportional selection '''
    # Make a population where proportional_selection has an obvious
    # reproducible choice
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    parents = Individual.evaluate_population(pop)
    # This selection operator will always select the [1, 1, 1] individual since
    # [0, 0, 0] has zero fitness
    selector = ops.proportional_selection(parents)

    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])

    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])


@pytest.mark.stochastic
def test_proportional_selection2():
    ''' Test of a stochastic proportional selection '''
    # Make a population where fitness proportional selection has an obvious
    # reproducible choice
    # Proportions here should be 1/4 and 3/4, respectively
    pop = [Individual([0, 1, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]
    # Assign a unique identifier to each individual
    pop[0].id = 0
    pop[1].id = 1

    # We first need to evaluate all the individuals so that
    # selection has fitnesses to compare
    pop = Individual.evaluate_population(pop)
    selected = ops.proportional_selection(pop)

    N = 1000
    p_thresh = 0.1
    observed_dist = statistical_helpers.collect_distribution(
        lambda: next(selected).id, samples=N)
    expected_dist = {pop[0].id: 0.25*N, pop[1].id: 0.75*N}
    print(f"Observed: {observed_dist}")
    print(f"Expected: {expected_dist}")
    assert(statistical_helpers.stochastic_equals(expected_dist,
                                                 observed_dist, p=p_thresh))


def test_proportional_selection_offset():
    ''' Test of proportional selection with a non-default offset '''
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    # evaluate population and negate fitness of second individual
    pop = Individual.evaluate_population(pop)
    pop[1].fitness = -pop[1].fitness

    # now we try to evaluate normally (this should throw a ValueError)
    # due to the negative fitness
    with pytest.raises(ValueError):
        selector = ops.proportional_selection(pop)
        selected = next(selector)
    # it should work by setting the offset to +3
    # this adds 3 to each fitness value, making the second
    # individual's fitness 0.
    selector = ops.proportional_selection(pop, offset=3)

    # we expect the first individual to always be selected
    # since the new zero point is now -3.
    selected = next(selector)
    assert np.all(selected.genome == [0, 0, 0])

    selected = next(selector)
    assert np.all(selected.genome == [0, 0, 0])


def test_proportional_selection_pop_min():
    ''' Test of proportional selection with pop-min offset '''
    # Create a population of positive fitness individuals
    # scaling the fitness by the population minimum makes it so the
    # least fit member never gets selected.
    pop = [Individual([0, 1, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    pop = Individual.evaluate_population(pop)

    selector = ops.proportional_selection(pop, offset='pop-min')

    # we expect that the second individual is always selected
    # since the new zero point will be at the minimum fitness
    # of the population
    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])

    selected = next(selector)
    assert np.all(selected.genome == [1, 1, 1])


def test_proportional_selection_custom_key():
    ''' Test of proportional selection with custom evaluation '''
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    def custom_key(individual):
        ''' Returns fitness based on MaxZeros '''
        return np.count_nonzero(individual.genome == 0)

    pop = Individual.evaluate_population(pop)
    selector = ops.proportional_selection(pop, key=custom_key)

    # we expect the first individual to always be selected
    # since its genome is all 0s
    selected = next(selector)
    assert np.all(selected.genome == [0, 0, 0])

    selected = next(selector)
    assert np.all(selected.genome == [0, 0, 0])


##############################
# Tests for naive_cyclic_selection()
##############################
def test_naive_cyclic_selection():
    """ Test of the naive deterministic cyclic selection """
    pop = [Individual([0, 0], problem=MaxOnes()),
           Individual([0, 1], problem=MaxOnes())]

    # This selection operator will deterministically cycle through the
    # given population
    selector = ops.naive_cyclic_selection(pop)

    selected = next(selector)
    assert np.all(selected.genome == [0, 0])

    selected = next(selector)
    assert np.all(selected.genome == [0, 1])

    # And now we cycle back to the first individual
    selected = next(selector)
    assert np.all(selected.genome == [0, 0])


##############################
# Tests for cyclic_selection()
##############################
def test_cyclic_selection():
    """ Test of the deterministic cyclic selection """

    # Set seed so that we get consistent test results.  I.e., it is possible
    # by happenstance for some tests to fail even though they're actually ok.
    # E.g., the cyclic selection tests will test if the test_sequence
    # shuffles between a complete cycle, but there's a chance that the same
    # test_sequence may come up in the random shuffle, so the test will fail.
    # However, if we set a random seed ahead of time, then we can control for
    # those pathological scenarios.
    random.seed(123)

    # We're just going to use integers for the population as that's
    # sufficient for testing this selection operator; we don't want to get in
    # the weeds with comparing individuals for test_sequence equivalency
    # testing.
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


##############################
# Tests for truncation_selection()
##############################
def test_truncation_selection():
    """ Basic truncation selection test"""
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([0, 0, 1], problem=MaxOnes()),
           Individual([1, 1, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]

    # We first need to evaluate all the individuals so that truncation
    # selection has fitnesses to compare
    pop = Individual.evaluate_population(pop)

    truncated = ops.truncation_selection(pop, 2)

    assert len(truncated) == 2

    # Just to make sure, check that the two best individuals from the
    # original population are in the selected population
    assert pop[2] in truncated
    assert pop[3] in truncated


def test_truncation_parents_selection():
    """ Test (mu + lambda), i.e., parents competing with offspring

    Create parent and offspring populations such that each has an "best" individual that will be selected by
    truncation selection.
    """
    parents = [Individual([0, 0, 0], problem=MaxOnes()),
               Individual([1, 1, 0], problem=MaxOnes())]

    parents = Individual.evaluate_population(parents)

    offspring = [Individual([0, 0, 1], problem=MaxOnes()),
                 Individual([1, 1, 1], problem=MaxOnes())]
    offspring = Individual.evaluate_population(offspring)

    truncated = ops.truncation_selection(offspring, 2, parents=parents)

    assert len(truncated) == 2

    assert parents[1] in truncated
    assert offspring[1] in truncated


def test_truncation_selection_with_nan1():
    """If truncation selection encounters a NaN and non-NaN fitness
    while maximizing, the non-NaN wins.
    """
    # Make a population where binary tournament_selection has an obvious
    # reproducible choice
    problem = MaxOnes()
    pop = [Individual([0, 0, 0], problem=problem),
           Individual([1, 1, 1], problem=problem)]

    # We first need to evaluate all the individuals so that truncation
    # selection has fitnesses to compare
    pop = Individual.evaluate_population(pop)

    # Now set the "best" to NaN
    pop[1].fitness = nan

    best = ops.truncation_selection(pop, size=1)

    assert pop[0] == best[0]


def test_truncation_selection_with_nan2():
    """If truncation selection encounters a NaN and non-NaN fitness
    while minimizing, the non-NaN wins.
    """
    problem = SpheroidProblem(maximize=False)

    pop = []

    pop.append(Individual([0], problem=problem))
    pop.append(Individual([1], problem=problem))

    pop = Individual.evaluate_population(pop)

    # First *normal* selection should yield the 0 as the "best"
    best = ops.truncation_selection(pop, size=1)
    assert pop[0] == best[0]

    # But now let's set that best to a NaN, which *should* force the other
    # individual to be selected.
    pop[0].fitness = nan

    best = ops.truncation_selection(pop, size=1)
    assert pop[1] == best[0]


##############################
# Tests for tournament_selection()
##############################
@pytest.mark.stochastic
def test_tournament_selection1():
    """If there are just two individuals in the population, then binary tournament
    selection will select the better one with 75% probability."""
    # Make a population where binary tournament_selection has an obvious
    # reproducible choice
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]
    # Assign a unique identifier to each individual
    pop[0].id = 0
    pop[1].id = 1

    # We first need to evaluate all the individuals so that
    # selection has fitnesses to compare
    pop = Individual.evaluate_population(pop)
    selected = ops.tournament_selection(pop)

    N = 1000
    p_thresh = 0.1
    observed_dist = statistical_helpers.collect_distribution(lambda: next(selected).id, samples=N)
    expected_dist = { pop[0].id: 0.25*N, pop[1].id: 0.75*N } 
    print(f"Observed: {observed_dist}")
    print(f"Expected: {expected_dist}")
    assert(statistical_helpers.stochastic_equals(expected_dist, observed_dist, p=p_thresh))


@pytest.mark.stochastic
def test_tournament_selection2():
    """If there are just two individuals in the population, and we set select_worst=True,
    then binary tournament selection will select the worse one with 75% probability."""
    # Make a population where binary tournament_selection has an obvious
    # reproducible choice
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]
    # Assign a unique identifier to each individual
    pop[0].id = 0
    pop[1].id = 1

    # We first need to evaluate all the individuals so that
    # selection has fitnesses to compare
    pop = Individual.evaluate_population(pop)
    selected = ops.tournament_selection(pop, select_worst=True)

    N = 1000
    p_thresh = 0.1
    observed_dist = statistical_helpers.collect_distribution(lambda: next(selected).id, samples=N)
    expected_dist = { pop[0].id: 0.75*N, pop[1].id: 0.25*N } 
    print(f"Observed: {observed_dist}")
    print(f"Expected: {expected_dist}")
    assert(statistical_helpers.stochastic_equals(expected_dist, observed_dist, p=p_thresh))

def test_tournament_selection_indices():
    """If an empty list is provided to tournament selection, it should be populated with
    the index of the selected individual.
    
    If we select a second individual, the list should be cleared and populated with the 
    index of the second individual."""
    pop = test_population

    indices = []
    op = ops.tournament_selection(indices=indices)

    # Select an individual
    s = next(op(pop))
    # Ensure the returned index is correct
    assert(len(indices) == 1)
    idx = indices[0]
    assert(idx >= 0)
    assert(idx < len(pop))
    assert(pop[idx] is s)

    # Select another individual
    s = next(op(pop))
    # Ensure the returned index is correct
    assert(len(indices) == 1)
    idx = indices[0]
    assert(idx >= 0)
    assert(idx < len(pop))
    assert(pop[idx] is s)


##############################
# Tests for random_selection()
##############################
@pytest.mark.stochastic
def test_random_selection1():
    """If there are just two individuals in the population, then random
    selection will select the better one with 50% probability."""
    pop = [Individual([0, 0, 0], problem=MaxOnes()),
           Individual([1, 1, 1], problem=MaxOnes())]
    # Assign a unique identifier to each individual
    pop[0].id = 0
    pop[1].id = 1

    # We first need to evaluate all the individuals so that
    # selection has fitnesses to compare
    pop = Individual.evaluate_population(pop)
    selected = ops.random_selection(pop)

    N = 1000
    p_thresh = 0.1
    observed_dist = statistical_helpers.collect_distribution(lambda: next(selected).id, samples=N)
    expected_dist = { pop[0].id: 0.5*N, pop[1].id: 0.5*N } 
    print(f"Observed: {observed_dist}")
    print(f"Expected: {expected_dist}")
    assert(statistical_helpers.stochastic_equals(expected_dist, observed_dist, p=p_thresh))


def test_random_selection_indices():
    """If an empty list is provided to random selection, it should be populated with
    the index of the selected individual.
    
    If we select a second individual, the list should be cleared and populated with the 
    index of the second individual."""
    pop = test_population

    indices = []
    op = ops.random_selection(indices=indices)

    # Select an individual
    s = next(op(pop))
    # Ensure the returned index is correct
    assert(len(indices) == 1)
    idx = indices[0]
    assert(idx >= 0)
    assert(idx < len(pop))
    assert(pop[idx] is s)

    # Select another individual
    s = next(op(pop))
    # Ensure the returned index is correct
    assert(len(indices) == 1)
    idx = indices[0]
    assert(idx >= 0)
    assert(idx < len(pop))
    assert(pop[idx] is s)
