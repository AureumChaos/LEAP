"""
  Probes are pipeline operators to instrument state that passes through the pipeline
  such as populations or individuals.

  TODO Will have to sync with Siggy on updating his code to work with the new paradigm.
  Shouldn't be too hard, and use of callbacks for inc_generation() should help as well as use
  of context objects.
"""
import csv
import sys

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from toolz import curry

from leap import ops as op


##############################
# print_probe
##############################
@curry
def print_probe(population, probe, stream=sys.stdout, prefix=''):
    """ pipeline operator for printing the given populaiton

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
# BestSoFar probe
##############################
class BestSoFarProbe(op.Operator):
    def __init__(self, context, stream=sys.stdout, header=True):
        self.bsf = None
        self.context = context
        self.writer = csv.DictWriter(stream, fieldname=['step', 'bsf'])

    def __call__(self, next_individual):
        assert(next_individual is not None)
        assert('leap' in self.context)
        assert('generation' in self.context['leap'])

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
class PopFitnessStatsCSVProbe(op.Operator):
    def __init__(self, context, stream=sys.stdout, header=True):
        assert (stream is not None)
        assert (hasattr(stream, 'write'))
        assert (context is not None)

        self.stream = stream
        self.context = context
        self.bsf_ind = None
        if header:
            stream.write('step, bsf, mean_fitness, std_fitness, min_fitness, max_fitness\n')

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
    An operator that records the specified attributes for all the individuals (or just the best individual) in
    `population` in CSV-format to the specified stream.

    :param population: list of individuals to take measurements from
    :param context: an optional context
    :param attributes: list of attribute names to record, as found in individuals' `attributes` field
    :return: `(population, context)`, unmodified (this allows you to place probes directly in an operator pipeline)

    Individuals contain some build-in attributes (namely fitness, genome), and also a `dict` of additional custom
    attributes called, well, `attributes`.  This class allows you to log all of the above.

    Most often, you will want to record only the best individual in the population at each step, and you'll just want
    to know its fitness and genome.  You can do this with this class's boolean flags.  For example, here's how you'd
    record the best individual's fitness and genome to a `StringIO` file object:

    >>> import io
    >>> from leap.data import test_population
    >>> stream = io.StringIO()
    >>> probe = AttributesCSVProbe(stream, best_only=True, do_fitness=True, do_genome=True)
    >>> probe.set_step(100)
    >>> probe(test_population, context=None)
    (..., None)

    >>> print(stream.getvalue())
    step, fitness, genome
    100, 4, [0, 1, 1, 1, 1]
    <BLANKLINE>

    You could just as easily use standard streams like `sys.stdout` for the `stream` parameter.

    Another common use of this task is to record custom attributes that are stored on individuals in certain kinds of
    experiments.  Here's how you would record the values of `ind.attributes['foo']` and `ind.attributes['bar']` for
    every individual in the population:

    >>> stream = io.StringIO()
    >>> probe = AttributesCSVProbe(stream, attributes=['foo', 'bar'])
    >>> probe.set_step(100)
    >>> r = probe(test_population, context=None)
    >>> print(stream.getvalue())
    step, foo, bar
    100, GREEN, Colorless
    100, 15, green
    100, BLUE, ideas
    100, 72.81, sleep
    <BLANKLINE>
    """

    def __init__(self, context, stream=sys.stdout, attributes=(), header=True, do_fitness=False, do_genome=False):
        assert (stream is not None)
        assert (hasattr(stream, 'write'))
        assert (len(attributes) >= 0)
        self.stream = stream
        self.attributes = attributes
        self.context = context

        self.do_fitness = do_fitness
        self.do_genome = do_genome

        self.writer = csv.DictWriter(stream, fieldnames=['step'] + attributes + ['fitness', 'genome'])
        if header:
            self.writer.writeheader()

    def __call__(self, next_individual):
        assert (next_individual is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])

        ind = next(next_individual)

        csvrow = {'step': self.context['leap']['generation'],
                  'fitness': ind.fitness,
                  'genome': str(ind.genome)
                  }
        for attr in self.attributes:
            if attr not in ind.attributes:
                raise ValueError('Attribute "{0}" not found in individual "{1}".'.format(attr, ind.__repr__()))
            csvrow[attr] = ind.attributes[attr]

        self.writer.writerow(csvrow)

        yield ind


##############################
# Class PlotProbe
##############################
class PopulationPlotProbe:
    """
    Measure and plot a population's fitness trajectory (or some other scalar value).

    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will be created).
    :param function f: a function that takes a population and returns a `float` value to plot on the y-axis (the default
        function plots the best-of-generation individual's fitness).
    :param xlim: Bounds of the horizontal axis.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param int modulo: take and plot a measurement every `modulo` steps (default 1).

    Attach this probe to matplotlib :class:`Axes` and then insert it into an EA's operator pipeline.

    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        from leap.probe import PlotProbe
        plt.figure()  # Setup a figure to plot to
        plot_probe = PlotProbe(ylim=(0, 70), ax=plt.gca())

        # Create an algorithm that contains the probe in the operator pipeline
        from leap.example.simple_ea import simple_ea
        from leap import core, real, operate as op

        l = 10
        mutate_prob = 1/l
        pop_size = 5
        ea = simple_ea(evals=1000, pop_size=pop_size,
                       individual_cls=core.Individual,
                       decoder=core.IdentityDecoder(),
                       problem=real.Spheroid(maximize=False),
                       evaluate=op.evaluate,

                       initialize=real.create_real_value_sequence(
                           bounds=[[-5.12, 5.12]] * l
                       ),

                       step_notify_list=[plot_probe.set_step], # STEP NOTIFICATION: sets plot_probe's x-coordinate

                       pipeline=[
                           # PIPELINE: sets plot_probe's y-coordinate
                           plot_probe,
                           op.tournament(n=pop_size),
                           op.cloning,
                           op.mutate_gaussian(prob=mutate_prob, std=1.0)
                       ])
        list(ea);


    To get a live-updated plot that words like a real-time video of the EA's progress, use this probe in conjunction
    with the `%matplotlib notebook` magic for Jupyter Notebook (as opposed to `%matplotlib inline`, which only
    allows static plots).

    """

    def __init__(self, context, ax=None, f=lambda x: best_of_gen(x).fitness, xlim=(0, 100), ylim=(0, 1), modulo=1):

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
        assert(population is not None)
        assert('leap' in self.context)
        assert('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            self.x = np.append(self.x, step)
            self.y = np.append(self.y, self.f(population))
            line = self.ax.lines[0]
            line.set_xdata(self.x)
            line.set_ydata(self.y)
            self.__rescale_ax()
            self.ax.figure.canvas.draw()
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
    Measure and plot a scatterplot of the populations' location in a 2-D phenotype space.

    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will be created).
    :param ~leap.problem.Problem contours: a problem defining a 2-D fitness function (this will be used to draw fitness
        contours in the background of the scatterplot).
    :param xlim: Bounds of the horizontal axis.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param float granularity: Spacing of the grid to sample points along while drawing the fitness contours.
    :param int modulo: take and plot a measurement every `modulo` steps (default 1).

    Attach this probe to matplotlib :class:`Axes` and then insert it into an EA's operator pipeline to get a live
    fitness plot that updates every `modulo` steps.

    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        from leap.probe import PlotTrajectoryProbe
        from leap.example.simple_ea import simple_ea
        from leap import core, real, operate as op

        # The fitness landscape
        problem = real.CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])

        # If no axis is provided, a new figure will be created for the probe to write to
        trajectory_probe = PlotTrajectoryProbe(contours=problem, xlim=(0, 1), ylim=(0, 1), granularity=0.025)

        # Create an algorithm that contains the probe in the operator pipeline

        l = 10
        mutate_prob = 1/l
        pop_size = 10
        ea = simple_ea(evals=50, pop_size=pop_size,
                       individual_cls=core.Individual,
                       decoder=core.IdentityDecoder(),
                       problem=real.Spheroid(maximize=False),
                       evaluate=op.evaluate,

                       initialize=real.create_real_value_sequence(
                           bounds=[[0.4, 0.6]] * l
                       ),

                       pipeline=[
                           trajectory_probe,  # Insert the probe into the pipeline like so
                           op.tournament(n=pop_size),
                           op.cloning,
                           op.mutate_gaussian(prob=mutate_prob, std=0.1, hard_bounds=(0, 1))
                       ])
        list(ea);


    """

    def __init__(self, context, ax=None, xlim=(-5.12, 5.12), ylim=(-5.12, 5.12), contours=None, granularity=0.1, modulo=1):
        if ax is None:
            ax = plt.subplot(111)
        if contours:
            @np.vectorize
            def v_fun(x, y):
                return contours.evaluate([x, y])

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
        assert(population is not None)
        assert('leap' in self.context)
        assert('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            self.x = np.array([ind.decode()[0] for ind in population])
            self.y = np.array([ind.decode()[1] for ind in population])
            self.sc.set_offsets(np.c_[self.x, self.y])
            self.__rescale_ax()
            self.ax.figure.canvas.draw()
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

    >>> from leap import ops
    >>> from leap.data import test_population
    >>> pop, _ = op.evaluate(test_population)
    >>> print(best_of_gen(pop))
    [0, 1, 1, 1, 1]
    """
    assert (len(population) > 0)
    return max(population)
