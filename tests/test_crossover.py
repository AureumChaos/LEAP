"""
    Unit tests for crossover operators
"""
import itertools

import pytest
import numpy as np
from toolz import partition

from leap_ec.individual import Individual
import leap_ec.ops as ops
import leap_ec.statistical_helpers as stat


##############################
# Tests for uniform_crossover()
##############################
def test_uniform_crossover():
    """Ensure that crossover with 100% swap probability completely swaps the two genomes."""
    pop = [Individual(np.array([0, 0])),
           Individual(np.array([1, 1]))]

    # We need a cyclic generator because there are only two individuals in the population, and once the first two
    # are selected for uniform crossover, the next two parents are selected and crossed over.  The cyclic iterator
    # ensures we just select the same two individuals again even though the yield statements in the uniform
    # crossover operator are not invoked again.
    i = ops.naive_cyclic_selection(pop)

    # Do swap with 100% certainty, which will cause the two individuals' genomes to exchange values
    new_pop = list(itertools.islice(ops.UniformCrossover(p_swap=1.0)(i), 2))
    assert np.all(new_pop[0].genome == [1, 1])
    assert np.all(new_pop[1].genome == [0, 0])

    # Note because we didn't clone the selected individuals, *the original population was changed*.
    assert np.all(pop[0].genome == [1, 1])
    assert np.all(pop[1].genome == [0, 0])


def test_uniform_crossover_probability1():
    """If we perform uniform rossover with a probabilty of 0.0, then the individuals will always be unmodified.
    
    This test calls the crossover opererator, which is stochastic, but we haven't marked it as part of the 
    stochastic test suite because there is no chance of a false failure (i.e. a test that fails even when
    there is no fault in the code) in this case."""
    N = 20
    unmodified_count = 0

    for i in range(N):

        pop = [Individual(np.array([0, 0])),
               Individual(np.array([1, 1]))]
        i = ops.naive_cyclic_selection(pop)
        new_pop = list(
                itertools.islice(ops.UniformCrossover(p_xover=0.0)(i), 2))

        if np.all(new_pop[0].genome == [0, 0]) and np.all(
                new_pop[1].genome == [1, 1]):
            unmodified_count += 1

    assert (unmodified_count == N)


def test_uniform_crossover_lineage():
    """Any result of crossover between individuals should result in
    individuals with the combined parents of those input.
    
    This test calls uniform crossover on a few combinations of individuals to
    ensure that the parents uuid lists are built correctly.
    """

    parents = [Individual(np.array([0, 0])) for _ in range(4)]
    children = [p.clone() for p in parents]

    pair_cross = list(itertools.islice(
            ops.UniformCrossover(p_xover=1.0)(ops.naive_cyclic_selection(children)),
            4))

    parent_uuids = [{lp.uuid, rp.uuid} for lp, rp in partition(2, parents)]
    for (lc, rc), pu in zip(partition(2, pair_cross), parent_uuids):
        assert len(lc.parents) == 2, "Left child did not have 2 parents."
        assert lc.parents == pu, "Left child's parent set is incorrect."
        assert len(rc.parents) == 2, "Right child did not have 2 parents."
        assert rc.parents == pu, "Right child's parent set is incorrect."

    triplet_child, = itertools.islice(
            ops.UniformCrossover(p_xover=1.0)(ops.naive_cyclic_selection(
                    [pair_cross[0], parents[2].clone()])), 1
    )
    triplet_uuids = parent_uuids[0] | {parents[2].uuid}
    assert len(triplet_child.parents) == 3, \
        ("Crossing a crossed individual a second time did not result in 3 "
         "parents.")
    assert triplet_child.parents == triplet_uuids, \
        ("Crossing a crossed individual a second time did not produce the "
         "correct parents.")

    # Note: this is child 1 and child 3 being crossed, so neither was
    # in the triplet
    quadruplet_child, = itertools.islice(
            ops.UniformCrossover(p_xover=1.0)(
                    ops.naive_cyclic_selection(pair_cross[1::2])), 1
    )
    quadruplet_uuids = {p.uuid for p in parents}
    assert len(
            quadruplet_child.parents) == 4, \
        ("Crossing two crossed individuals a second time did not result in "
         "4 parents.")
    assert quadruplet_child.parents == quadruplet_uuids, \
        ("Crossing two crossed individuals a second time did not produce "
         "the correct parents.")


@pytest.mark.stochastic
def test_uniform_crossover_probability2():
    """ If we perform uniform crossover with a probabilty of 1.0, then we should
    see genes swapped by default with probability 0.2. """
    N = 5000
    observed_dist = {'Unmodified'        : 0, 'Only left swapped': 0,
                     'Only right swapped': 0, 'Both swapped': 0}

    # Run crossover N times on a fixed pair of two-gene individuals
    for i in range(N):

        pop = [Individual(np.array([0, 0])),
               Individual(np.array([1, 1]))]
        i = ops.naive_cyclic_selection(pop)
        new_pop = list(
                itertools.islice(ops.UniformCrossover(p_xover=1.0)(i), 2))

        # There are four possible outcomes, which we will count the occurence of
        if np.all(new_pop[0].genome == [0, 0]) and np.all(
                new_pop[1].genome == [1, 1]):
            observed_dist['Unmodified'] += 1
        elif np.all(new_pop[0].genome == [1, 0]) and np.all(
                new_pop[1].genome == [0, 1]):
            observed_dist['Only left swapped'] += 1
        elif np.all(new_pop[0].genome == [0, 1]) and np.all(
                new_pop[1].genome == [1, 0]):
            observed_dist['Only right swapped'] += 1
        elif np.all(new_pop[0].genome == [1, 1]) and np.all(
                new_pop[1].genome == [0, 0]):
            observed_dist['Both swapped'] += 1
        else:
            assert (False)

    assert (N == sum(observed_dist.values()))

    p = 0.01
    p_swap = 0.2
    # This is the count we expect to see of each combination
    # Each locus swaps with p_swap.
    expected_dist = {
        'Unmodified'        : int((1 - p_swap) * (1 - p_swap) * N),
        'Only left swapped' : int(p_swap * (1 - p_swap) * N),
        'Only right swapped': int((1 - p_swap) * p_swap * N),
        'Both swapped'      : int(p_swap ** 2 * N)
    }

    # Use a χ-squared test to see if our experiment matches what we expect
    assert (stat.stochastic_equals(expected_dist, observed_dist, p=p))


##############################
# Tests for n_ary_crossover()
##############################
def test_n_ary_crossover_bad_crossover_points():
    """ Test assertions for having more crossover points than genome length """
    pop = [Individual(np.array([0, 0])),
           Individual(np.array([1, 1]))]

    i = ops.naive_cyclic_selection(pop)

    with pytest.raises(RuntimeError):
        new_pop = list(itertools.islice(ops.NAryCrossover(num_points=3)(i), 2))


def test_n_ary_crossover():
    """ If we crossover two individuals with two bits each, the children
    should either be swapped copies of their parents, or they should exchange
    the second bit and keep the first bit unmodified.
    """
    pop = [Individual(np.array([0, 0])),
           Individual(np.array([1, 1]))]

    i = ops.naive_cyclic_selection(pop)

    new_pop = list(itertools.islice(ops.NAryCrossover(num_points=1)(i), 2))

    # Given that there are only two genes, one [0,0] and the other [1,1] and a single crossover point, and that the
    # only two valid crossover points are 0 or 1, then there are two possible valid states for offspring with single
    # point crossover.
    assert np.all(pop[0].genome == [1, 1]) or np.all(pop[0].genome == [0, 1])
    assert np.all(pop[1].genome == [0, 0]) or np.all(pop[1].genome == [1, 0])


@pytest.mark.stochastic
def test_n_ary_crossover_probability():
    """ If we perform crossover with a probabilty of 0.5, then the individuals
    will be unmodified 50% of the time.
    """
    N = 5000
    unmodified_count = 0

    for i in range(N):

        pop = [Individual(np.array([0, 0])),
               Individual(np.array([1, 1]))]
        i = ops.naive_cyclic_selection(pop)
        new_pop = list(
                itertools.islice(
                    ops.NAryCrossover(num_points=1, p_xover=0.5)(i),
                    2))

        if np.all(new_pop[0].genome == [0, 0]) and np.all(
                new_pop[1].genome == [1, 1]):
            unmodified_count += 1

    p = 0.01
    observed_dist = {'Unmodified': unmodified_count,
                     'Modified'  : N - unmodified_count}
    assert (stat.equals_uniform(observed_dist, p=p))
