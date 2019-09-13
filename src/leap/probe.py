import sys

from toolz import curry

from leap.core import Individual
from leap.operate import Operator


@curry
def print_probe(population, context, probe, stream=sys.stdout, prefix=''):
    val = prefix + str(probe(population, context))
    stream.write(val)
    return population, context


##############################
# Class MemoryProbe
##############################
class MemoryProbe(Operator):
    def __init__(self, probe):
        self.probe = probe
        self.data = []

    def __call__(self, population, context):
        self.data.append(self.probe(population, context))
        return population, context

    def clear(self):
        self.data = []


##############################
# best_of_gen  probe
##############################
def best_of_gen(population, context=None):
    """
    Syntactic sugar to select the best individual in a population.

    :param population: a list of individuals
    :param context: optional `dict` of auxiliary state (ignored)

    >>> from leap import core, binary
    >>> from leap import operate as op
    >>> pop = [Individual([1, 0, 1, 1, 0], core.IdentityDecoder(), binary.MaxOnes()), \
               Individual([0, 0, 1, 0, 0], core.IdentityDecoder(), binary.MaxOnes()), \
               Individual([0, 1, 1, 1, 1], core.IdentityDecoder(), binary.MaxOnes()), \
               Individual([1, 0, 0, 0, 1], core.IdentityDecoder(), binary.MaxOnes())]
    >>> pop, _ = op.evaluate(pop)
    >>> pop, _ = best_of_gen(pop)
    >>> print(pop)
    [0, 1, 1, 1, 1]
    """
    assert(len(population) > 0)
    return max(population), context


##############################
# BestSoFar probe
##############################
class BestSoFar(Operator):
    def __init__(self, just_fitness=False):
        self.bsf = None
        self.just_fitness = just_fitness

    def __call__(self, population, context):
        for ind in population:
            if self.bsf is None or (ind > self.bsf):
                self.bsf = ind

        return self.bsf.fitness if self.just_fitness else self.bsf, context
