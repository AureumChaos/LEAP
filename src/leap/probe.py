import sys

import numpy as np
from toolz import curry

from leap import operate as op


##############################
# print_probe
##############################
@curry
def print_probe(population, context, probe, stream=sys.stdout, prefix=''):
    val = prefix + str(probe(population, context))
    stream.write(val)
    return population, context


##############################
# Class CSVFitnessStatsProbe
##############################
class CSVFitnessStatsProbe(op.Operator):
    def __init__(self, stream, header=True):
        assert(stream is not None)
        assert(hasattr(stream, 'write'))
        self.stream = stream
        self.bsf_ind = None
        self.step = None
        if header:
            stream.write('step, bsf, mean_fitness, std_fitness, min_fitness, max_fitness\n')

    def set_step(self, step):
        """Have your algorithm call this method every time the generation or step changes, so that the probe knows what
        step number to record in the CSV."""
        assert(step >= 0)
        self.step = step

    def __call__(self, population, context):
        assert (population is not None)
        self.stream.write(str(self.step) + ', ')

        best_ind = best_of_gen(population)
        if self.bsf_ind is None or (best_ind > self.bsf_ind):
            self.bsf_ind = best_ind
        self.stream.write(str(self.bsf_ind.fitness) + ', ')

        fitnesses = [x.fitness for x in population]
        self.stream.write(str(np.mean(fitnesses)) + ', ')
        self.stream.write(str(np.std(fitnesses)) + ', ')
        self.stream.write(str(np.min(fitnesses)) + ', ')
        self.stream.write(str(np.max(fitnesses)))
        self.stream.write('\n')
        return population, context


##############################
# Class CSVAttributesProbe
##############################
class CSVAttributesProbe(op.Operator):
    """
    An operator that records the specified attributes for all the individuals in `population` in CSV-format to the
    specified stream.

    :param population: list of individuals to take measurements from
    :param context: an optional context
    :param attributes: list of attribute names to record, as found in individuals' `attributes` field
    :return: `(population, context)`, unmodified (this allows you to place probes directly in an operator pipeline)

    Here's an example that writes CSV output to a `StringIO` file object.  You can also use standard streams like files
    or `sys.stdout`:

    >>> import io
    >>> from leap.data import test_population
    >>> stream = io.StringIO()
    >>> probe = CSVAttributesProbe(stream, ['foo', 'bar'])
    >>> probe.set_step(100)
    >>> probe(test_population, context=None)
    (..., None)

    >>> print(stream.getvalue())
    step, foo, bar
    100, GREEN, Colorless
    100, 15, green
    100, BLUE, ideas
    100, 72.81, sleep
    <BLANKLINE>
    """

    def __init__(self, stream, attributes, header=True, do_fitness=True, do_genome=False):
        assert(stream is not None)
        assert(hasattr(stream, 'write'))
        assert(len(attributes) > 0)
        self.stream = stream
        self.attributes = attributes
        self.step = None
        self.do_fitness = do_fitness
        self.do_genome = do_genome
        if header:
            stream.write('step, ' + ', '.join(attributes) + '\n')

    def set_step(self, step):
        """Have your algorithm call this method every time the generation or step changes, so that the probe knows what
        step number to record in the CSV."""
        assert(step >= 0)
        self.step = step

    def __call__(self, population, context):
        # TODO Need to quote attribute values properly in case they contain delimiters
        # TODO Replace this custom code with calls to a CSV package?
        # TODO would be nice if we defaulted to checking a context variable if set_step() is never called
        assert(population is not None)
        for ind in population:
            self.stream.write(str(self.step))
            for attr in self.attributes:
                if attr not in ind.attributes:
                    raise ValueError('Attribute "{0}" not found in individual "{1}".'.format(attr, ind.__repr__()))
                self.stream.write(', ' + str(ind.attributes[attr]))
            self.stream.write('\n')
        return population, context


##############################
# Class MemoryProbe
##############################
class MemoryProbe(op.Operator):
    def __init__(self, probe):
        assert(probe is not None)
        assert(callable(probe))
        self.probe = probe
        self.data = []

    def __call__(self, population, context):
        assert(population is not None)
        self.data.append(self.probe(population, context))
        return population, context

    def clear(self):
        self.data = []


##############################
# best_of_gen function
##############################
def best_of_gen(population):
    """
    Syntactic sugar to select the best individual in a population.

    :param population: a list of individuals
    :param context: optional `dict` of auxiliary state (ignored)

    >>> from leap import operate as op
    >>> from leap.data import test_population
    >>> pop, _ = op.evaluate(test_population)
    >>> print(best_of_gen(pop))
    [0, 1, 1, 1, 1]
    """
    assert(len(population) > 0)
    return max(population)


##############################
# BestSoFar probe
##############################
class BestSoFar(op.Operator):
    def __init__(self, just_fitness=False):
        self.bsf = None
        self.just_fitness = just_fitness

    def __call__(self, population, context):
        for ind in population:
            if self.bsf is None or (ind > self.bsf):
                self.bsf = ind

        return self.bsf.fitness if self.just_fitness else self.bsf, context
