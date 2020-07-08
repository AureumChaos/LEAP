"""Probes are pipeline operators to instrument state that passes through the
pipeline such as populations or individuals. """
import csv
import sys

from typing import Iterator

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from toolz import curry

from leap_ec import ops as op
from leap_ec.ops import iteriter_op


##############################
# print_probe
##############################
@curry
def print_probe(population, probe, stream=sys.stdout, prefix=''):
    """ pipeline operator for printing the given population

    :param population:
    :param context:
    :param probe:
    :param stream:
    :param prefix:
    :return:
    """
    val = prefix + str(probe(population))
    stream.write(val)
    return population


##############################
# print_individual
##############################
@curry
@iteriter_op
def print_individual(next_individual: Iterator, prefix='', stream=sys.stdout) -> Iterator:
    """ Just echoes the individual from within the pipeline

    Uses next_individual.__str__

    :param next_individual: iterator for next individual to be printed
    :return: the same individual, unchanged
    """

    while True:
        individual = next(next_individual)

        print(f'{prefix}{individual!s}', file=stream)

        yield individual


##############################
# BestSoFar probe
##############################
class BestSoFarProbe(op.Operator):
    def __init__(self, context, stream=sys.stdout, header=True):
        self.bsf = None
        self.context = context
        self.writer = csv.DictWriter(stream, fieldnames=['step', 'bsf'])

    def __call__(self, next_individual):
        assert (next_individual is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])

        ind = next(next_individual)
        if self.bsf is None or (ind > self.bsf):
            self.bsf = ind

        self.writer.writerow({'step': self.context['leap']['generation'],
                              'bsf': self.bsf.fitness
                              })

        yield ind


##############################
# Class PopFitnessStatsProbe
##############################
class FitnessStatsCSVProbe(op.Operator):
    def __init__(self, context, stream=sys.stdout, header=True):
        assert (stream is not None)
        assert (hasattr(stream, 'write'))
        assert (context is not None)

        self.stream = stream
        self.context = context
        self.bsf_ind = None
        if header:
            stream.write(
                'step, bsf, mean_fitness, std_fitness, min_fitness, max_fitness\n')

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])

        self.stream.write(str(self.context['leap']['generation']) + ', ')

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
        return population


##############################
# Class AttributesCSVProbe
##############################
class AttributesCSVProbe(op.Operator):
    """
    An operator that records the specified attributes for all the individuals
    (or just the best individual) in `population` in CSV-format to the
    specified stream.

    :param population: list of individuals to take measurements from
    :param context: an optional context
    :param attributes: list of attribute names to record, as found in
        individuals' `attributes` field

    :return: `(population, context)`, unmodified (this allows you to place
        probes directly in an operator pipeline)

    Individuals contain some build-in attributes (namely fitness, genome),
    and also a `dict` of additional custom attributes called, well,
    `attributes`.  This class allows you to log all of the above.

    Most often, you will want to record only the best individual in the
    population at each step, and you'll just want to know its fitness and
    genome.  You can do this with this class's boolean flags.  For example,
    here's how you'd record the best individual's fitness and genome to a
    dataframe:

    >>> from leap_ec import core
    >>> from leap_ec.data import test_population
    >>> probe = AttributesCSVProbe(core.context, do_dataframe=True, best_only=True, do_fitness=True, do_genome=True)
    >>> core.context['leap']['generation'] = 100
    >>> probe(test_population) == test_population
    True

    You can retrieve the result programatically from the `dataframe` property:

    >>> probe.dataframe
       step  fitness           genome
    0   100        4  [0, 1, 1, 1, 1]

    By default, the results are also written to `sys.stdout`.  You can pass
    any file object you like into the `stream` parameter.

    Another common use of this task is to record custom attributes that are
    stored on individuals in certain kinds of experiments.  Here's how you
    would record the values of `ind.foo` and `ind.bar` for every individual
    in the population.  We write to a stream object this time to demonstrate
    how to use the probe without a dataframe:

    >>> import io
    >>> stream = io.StringIO()
    >>> probe = AttributesCSVProbe(core.context, attributes=['foo', 'bar'], stream=stream)
    >>> core.context['leap']['generation'] = 100
    >>> r = probe(test_population)
    >>> print(stream.getvalue())
    step,foo,bar
    100,GREEN,Colorless
    100,15,green
    100,BLUE,ideas
    100,72.81,sleep
    <BLANKLINE>
    """

    def __init__(self, context, attributes=(), stream=sys.stdout, do_dataframe=False, best_only=False, header=True, do_fitness=False,
                 do_genome=False, notes={}, computed_columns={}, job=None):
        assert ((stream is None) or hasattr(stream, 'write'))
        assert (len(attributes) >= 0)
        self.context = context
        self.stream = stream
        self.attributes = attributes
        self.best_only = best_only

        self.do_fitness = do_fitness
        self.do_genome = do_genome
        self.notes = notes
        self.computed_columns = computed_columns
        self.job = job
        self.do_dataframe = do_dataframe

        if (not do_dataframe) and stream is None:
            raise ValueError(
                "Both 'stream'=None and 'do_dataframe'=False, but at least one must be enabled.")

        fieldnames = ['step'] + list(attributes)
        if job:
            fieldnames.append('job')
        for name in notes.keys():
            fieldnames.append(name)
        if do_fitness:
            fieldnames.append('fitness')
        if do_genome:
            fieldnames.append('genome')
        for name in computed_columns.keys():
            fieldnames.append(name)

        self.fieldnames = fieldnames

        if self.do_dataframe:
            # We'll store rows of data as dicts in this list as we collect them
            self.data = []

        self.writer = None
        if stream is not None:
            # We'll write rows of data to this stream as we collect them
            self.writer = csv.DictWriter(
                stream, fieldnames=fieldnames, lineterminator='\n')
            if header:
                self.writer.writeheader()

    @property
    def dataframe(self):
        """Property for retrieving a Pandas DataFrame representation of the
        collected data. """
        if not self.do_dataframe:
            raise ValueError('Tried to retrieve a dataframe of results, but this ' +
                             f'{type(AttributesCSVProbe).__name__} was initialized with dataframe=False.')
        # We create the DataFrame on demand because it's inefficient to append to a DataFrame,
        # so we only want to create it after we are done generating data.
        return pd.DataFrame(self.data, columns=self.fieldnames)

    def __call__(self, population):
        """When called (i.e. as part of an operator pipeline), take a
        population of individuals and collect data from it. """
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])

        individuals = [max(population)] if self.best_only else population

        for ind in individuals:
            row = self.get_row_dict(ind)
            if self.writer is not None:
                self.writer.writerow(row)

            if self.do_dataframe:
                self.data.append(row)

        return population

    def get_row_dict(self, ind):
        """Compute a full row of data from a given individual."""
        row = {'step': self.context['leap']['generation']}

        for attr in self.attributes:
            if attr not in ind.__dict__:
                raise ValueError(
                    'Attribute "{0}" not found in individual "{1}".'.format(
                        attr, ind.__repr__()))
            row[attr] = ind.__dict__[attr]

        if self.job:
            row['job'] = self.job
        for k, v in self.notes.items():
            row[k] = v
        if self.do_fitness:
            row['fitness'] = ind.fitness
        if self.do_genome:
            row['genome'] = str(ind.genome)
        for k, f in self.computed_columns.items():
            row[k] = f(row)

        return row


##############################
# Class PopulationPlotProbe
##############################
class PopulationPlotProbe:
    """
    Measure and plot a population's fitness trajectory (or some other scalar
    value).

    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
        be created).
    :param function f: a function that takes a population and returns a
        `float` value to plot on the y-axis (the default function plots the
        best-of-generation individual's fitness).
    :param xlim: Bounds of the horizontal axis.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param int modulo: take and plot a measurement every `modulo` steps (
        default 1).

    Attach this probe to matplotlib :class:`Axes` and then insert it into an
    EA's operator pipeline.

    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        from leap_ec import core
        from leap_ec.probe import PopulationPlotProbe


        plt.figure()  # Setup a figure to plot to
        plot_probe = PopulationPlotProbe(core.context, ylim=(0, 70), ax=plt.gca())


        # Create an algorithm that contains the probe in the operator pipeline
        from leap_ec import ops, real_problems
        from leap_ec.algorithm import generational_ea

        l = 10
        pop_size = 10
        ea = generational_ea(generations=100, pop_size=pop_size,
                             problem=real_problems.SpheroidProblem(maximize=False),

                             representation=core.Representation(
                                individual_cls=core.Individual,
                                decoder=core.IdentityDecoder(),
                                initialize=core.create_real_vector(bounds=[[-5.12, 5.12]] * l)
                             ),

                             pipeline=[
                                 plot_probe,  # Insert the probe into the pipeline like so
                                 ops.tournament,
                                 ops.clone,
                                 ops.mutate_gaussian(std=1.0),
                                 ops.evaluate,
                                 ops.pool(size=pop_size)
                             ])
        list(ea);



    To get a live-updated plot that words like a real-time video of the EA's
    progress, use this probe in conjunction with the `%matplotlib notebook`
    magic for Jupyter Notebook (as opposed to `%matplotlib inline`,
    which only allows static plots).

    """

    def __init__(self, context, ax=None, f=lambda x: best_of_gen(
            x).fitness, xlim=(0, 100), ylim=(0, 1), modulo=1):

        if ax is None:
            ax = plt.subplot(111)
        ax.plot([], [])
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        self.ax = ax
        self.left, self.right = xlim
        self.bottom, self.top = ylim
        self.f = f
        self.x = np.array([])
        self.y = np.array([])
        self.modulo = modulo
        self.context = context

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            self.x = np.append(self.x, step)
            self.y = np.append(self.y, self.f(population))
            line = self.ax.lines[0]
            line.set_xdata(self.x)
            line.set_ydata(self.y)
            self.__rescale_ax()
            self.ax.figure.canvas.draw()
            plt.pause(0.000001)
        return population

    def __rescale_ax(self):
        if np.min(self.x) < self.left:
            self.ax.set_xlim(left=np.min(self.x))
        if np.max(self.x) > self.right:
            self.ax.set_xlim(right=np.max(self.x))
        if np.min(self.y) < self.bottom:
            self.ax.set_ylim(bottom=np.min(self.y))
        if np.max(self.y) > self.top:
            self.ax.set_ylim(top=np.max(self.y))


##############################
# Class PopTrajectoryProbe
##############################
class PlotTrajectoryProbe:
    """
    Measure and plot a scatterplot of the populations' location in a 2-D
    phenotype space.

    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
        be created).
    :param xlim: Bounds of the horizontal axis.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)

    :param ~leap.problem.Problem contours: a problem defining a 2-D fitness
    function (this will be used to draw fitness contours in the background of
    the scatterplot).

    :param float granularity: (Optional) spacing of the grid to sample points
        along while drawing the fitness contours. If none is given, then the
        granularity will default to 1/50th of the range of the function's
        `bounds` attribute.
    :param int modulo: take and plot a measurement every `modulo` steps (
        default 1).

    Attach this probe to matplotlib :class:`Axes` and then insert it into an
    EA's operator pipeline to get a live fitness plot that updates every
    `modulo` steps.

    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        from leap_ec.probe import PlotTrajectoryProbe
        from leap_ec.algorithm import generational_ea
        from leap_ec import core, ops, real_problems

        # The fitness landscape
        problem = real_problems.CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])

        # If no axis is provided, a new figure will be created for the probe to write to
        trajectory_probe = PlotTrajectoryProbe(context=core.context,
                                               contours=problem,
                                               xlim=(0, 1), ylim=(0, 1),
                                               granularity=0.025)

        # Create an algorithm that contains the probe in the operator pipeline

        pop_size = 100
        ea = generational_ea(generations=20, pop_size=pop_size,
                             problem=problem,

                             representation=core.Representation(
                                individual_cls=core.Individual,
                                initialize=core.create_real_vector(bounds=[[0.4, 0.6]] * 2),
                                decoder=core.IdentityDecoder()
                             ),

                             pipeline=[
                                 trajectory_probe,  # Insert the probe into the pipeline like so
                                 ops.tournament,
                                 ops.clone,
                                 ops.mutate_gaussian(std=0.1, hard_bounds=(0, 1)),
                                 ops.evaluate,
                                 ops.pool(size=pop_size)
                             ])
        list(ea);


    """

    def __init__(self, context, ax=None, xlim=(-5.12, 5.12), ylim=(-5.12, 5.12),
                 contours=None, granularity=None,
                 modulo=1):
        if ax is None:
            ax = plt.subplot(111)
        if contours:
            @np.vectorize
            def v_fun(x, y):
                return contours.evaluate([x, y])

            if granularity is None:
                granularity = (contours.bounds[1] - contours.bounds[0]) / 50.
            x = np.arange(xlim[0], xlim[1], granularity)
            y = np.arange(ylim[0], ylim[1], granularity)
            xx, yy = np.meshgrid(x, y)
            ax.contour(xx, yy, v_fun(xx, yy))

        self.sc = ax.scatter([], [])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.ax = ax
        self.left, self.right = xlim
        self.bottom, self.top = ylim
        self.x = np.array([])
        self.y = np.array([])
        self.modulo = modulo
        self.context = context

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            self.x = np.array([ind.decode()[0] for ind in population])
            self.y = np.array([ind.decode()[1] for ind in population])
            self.sc.set_offsets(np.c_[self.x, self.y])
            self.__rescale_ax()
            self.ax.figure.canvas.draw()
            plt.pause(0.000001)
        return population

    def __rescale_ax(self):
        if np.min(self.x) < self.left:
            self.ax.set_xlim(left=np.min(self.x))
        if np.max(self.x) > self.right:
            self.ax.set_xlim(right=np.max(self.x))
        if np.min(self.y) < self.bottom:
            self.ax.set_ylim(bottom=np.min(self.y))
        if np.max(self.y) > self.top:
            self.ax.set_ylim(top=np.max(self.y))


##############################
# best_of_gen function
##############################
def best_of_gen(population):
    """
    Syntactic sugar to select the best individual in a population.

    :param population: a list of individuals
    :param context: optional `dict` of auxiliary state (ignored)

    >>> from leap_ec import core, ops
    >>> from leap_ec.data import test_population
    >>> print(best_of_gen(test_population))
    [0, 1, 1, 1, 1]
    """
    assert (len(population) > 0)
    return max(population)
