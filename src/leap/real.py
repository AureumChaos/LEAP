import random

import numpy as np

from leap.problem import ScalarProblem
from leap.individual import Individual


##############################
# Closure real-genome initializer
##############################
def initialize_vectors(decoder, problem, bounds):
    """

    :param decoder:
    :param problem:
    :param bounds:
    :return:

    >>> from leap import decode, real
    >>> bounds = [(0, 1), (0, 1), (-1, 100)]
    >>> init = initialize_vectors(decode.IdentityDecoder(), real.Spheroid(), bounds)
    >>> for x in init(5):
    ...     print(x) # +doctest: ELLIPSIS
    [...]
    [...]
    [...]
    [...]
    [...]
    """
    def generate_genome():
        for (min, max) in bounds:
            yield random.uniform(min, max)

    def f(pop_size):
        return [Individual(list(generate_genome()), decoder, problem) for _ in range(pop_size)]

    return f


##############################
# Class Spheroid
##############################
class Spheroid(ScalarProblem):
    """ Classic spheroid problem

    """

    """ Standard bounds for a sphere functions solution."""
    bounds = [(-5.12, 5.12)]*10

    # TODO See if we get an error if we try to add a constructor that doesn't set maximize

    def __init__(self, maximize=True):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the spheroid function from a real-valued list phenome:

        >>> phenome = [0.5, 0.8, 1.5]
        >>> Spheroid().evaluate(phenome)
        3.14

        :param phenome: to be evaluated
        :return: sum(phenome**2)
        """
        return sum([x**2 for x in phenome])

    def worse_than(self, first_fitness, second_fitness):
        """
        We maximize by default:

        >>> from leap import decode
        >>> s = Spheroid()
        >>> s.worse_than(100, 10)
        False

        >>> from leap import decode
        >>> s = Spheroid(maximize=False)
        >>> s.worse_than(100, 10)
        True
        """
        return super().worse_than(first_fitness, second_fitness)


##############################
# Class CosineFamilyProblem
##############################
class CosineFamilyProblem(ScalarProblem):
    def __init__(self, alpha, global_optima_counts, local_optima_counts, maximize=True):
        super().__init__(maximize)
        self.alpha = alpha
        self.dimensions = len(global_optima_counts)
        assert(len(local_optima_counts) == self.dimensions)
        self.global_optima_counts = np.array(global_optima_counts)
        self.local_optima_counts = np.array(local_optima_counts)

    def evaluate(self, phenome):
        phenome = np.array(phenome)
        term1 = -np.cos((self.global_optima_counts - 1) * 2 * np.pi * phenome)
        term2 = - self.alpha * np.cos((self.global_optima_counts - 1) * 2 * np.pi * self.local_optima_counts * phenome)
        value = np.sum(term1 + term2)/(2*self.dimensions)
        # We modify the original function to make it a maximization problem
        # and so that the global optima are scaled to always have a fitness of 1
        return -2/(self.alpha + 1) * value
