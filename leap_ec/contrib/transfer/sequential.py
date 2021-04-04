"""
Experimental algorithms for sequential evolutionary transfer.

This module provides general mechanisms that allow an algorithm to
learn from experience on past problems, and to reuse that experience
on future problems.
"""

import abc
import csv


class Repertoire(abc.ABC):
    """Abstract definition of a 'repertoire' algorithm for evolutionary transfer.

    A repertoire is a wrapper for an algorithm that can be trained on a set of problems,
    from which is learns and encodes some form of memory, which can be applied to new
    problems.
    """

    @abc.abstractmethod
    def build_repertoire(self, problems, initialize, algorithm):
        """Train the repertoire on a set of problems.

        :param problems: a list of Problems to train on.
        :param initialize: a function that generates a population.
        :param algorithm: an algorithm function, which may be parameterized with an initialize function."""
        # XXX The only reason this takes an 'initialize' function is because we often implement repertoires via
        #     population seeding.  This should be generalized—the Repertoire should parameterize an algorithm in
        #     some way, but it may or may not do so by biasing the initialization of the popularion.
        pass

    @abc.abstractmethod
    def apply(self, problem, algorithm):
        """Apply the repertoire to a new problem.

        :param problem: the Problem to solve.
        :param algorithm: the algorithm to apply.
        """

        pass


class PopulationSeedingRepertoire:
    """ A repertoire method that works by seeding the population with
    individuals that  were successful on past problems.

    This works by injecting an `initialize` function into the wrapped
    algorithm's parameterization.  During training, we inject a
    standard initializer (i.e. that create a random population), but
    when applying the repertoire, we use a special initializer that
    draws individuals from the repertoire's memory.

    :param initialize: a standard initializer to create random populations during training.
    :param algorithm: the wrapped algorithm, which should take an initialize argument.
    :param repfile: an optional path to save the repertoire's memory to.
    """
    def __init__(self, initialize, algorithm, repfile=None):
        assert(algorithm is not None)
        if repfile:
            with open(repfile, 'r') as f:
                self.repertoire = list(csv.reader(
                    f, quoting=csv.QUOTE_NONNUMERIC))
        else:
            self.repertoire = []
        self.initialize = initialize
        self.algorithm = algorithm

    def build_repertoire(self, problems, problem_kwargs):
        """Train the repertoire on a set of problems.

        The best solution found on each problem will be saved into the repertoire.
        """
        assert(problems is not None)
        assert(len(problems) >= 0)
        assert(problem_kwargs is None or len(problem_kwargs) == len(problems))
        if problem_kwargs is None:
            problem_kwargs = [{}] * len(problems)
        results = [
            self.algorithm(
                p,
                self.initialize,
                **problem_kwargs[i]) for i,
            p in enumerate(problems)]
        # Execute each algorithm sequentially
        results = [list(ea) for ea in results]
        assert(len(results) == len(problems))
        for r in results:
            last_step, last_ind = r[-1]
            self.repertoire.append(last_ind.genome)

    def export(self, path):
        """Write the repertoire of saved individuals out to a CSV file."""
        with open(path, 'w') as f:
            csv.writer(f).writerows(self.repertoire)

    def apply(self, problem, **kwargs):
        """Solve a new problem by injecting the all the individuals from the
        repertoire into the new initial population."""
        repertoire_init = initialize_seeded(self.initialize, self.repertoire)
        return self.algorithm(problem, repertoire_init, **kwargs)


def initialize_seeded(initialize, seed_pop):
    """A population initializer that injects a fixed list of seed individuals
    into the population, and fills the remaining space with newly generated
    individuals.

    >>> import numpy as np
    >>> from leap_ec.real_rep.initializers import create_real_vector
    >>> random_init = create_real_vector(bounds=[[0, 0]] * 2)
    >>> init = initialize_seeded(random_init, [np.array([5.0, 5.0]), np.array([4.5, -6])])
    >>> [init() for _ in range(5)]
    [array([5., 5.]), array([ 4.5, -6. ]), array([0., 0.]), array([0., 0.]), array([0., 0.])]

    """
    assert (initialize is not None)
    assert (seed_pop is not None)

    i = 0

    def create():
        nonlocal i
        if i < len(seed_pop):
            ind = seed_pop[i]
            i += 1
            return ind
        else:
            return initialize()

    return create
