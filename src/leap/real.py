from leap.problem import ScalarProblem
from leap.individual import Individual
import random


##############################
# Closure real-genome initializer
##############################
def initialize_vectors(pop_size, decoder, problem, *bounds):
    """

    :param pop_size:
    :param decoder:
    :param problem:
    :param bounds:
    :return:

    >>> from leap import decode, real
    >>> bounds = [(0, 1), (0, 1), (-1, 100)]
    >>> init = initialize_vectors(5, decode.IdentityDecoder(), real.Spheroid(), *bounds)
    >>> for x in init():
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

    def f():
        return [Individual(list(generate_genome()), decoder, problem) for _ in range(pop_size)]

    return f



##############################
# Class Spheroid
##############################
class Spheroid(ScalarProblem):
    """ Classic spheroid problem
    """
    def __init__(self, maximize=True):
        super().__init__(maximize)

    def evaluate(self, individual):
        """

        :param individual: to be evaluated
        :return: sum(individual.genome**2)
        """
        return sum([x**2 for x in individual.decode()])
