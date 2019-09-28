import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from leap.problem import ScalarProblem


##############################
# Closure real-genome initializer
##############################
def initialize_vectors_uniform(bounds):
    """

    :param decoder:
    :param problem:
    :param bounds:
    :return:

    >>> from leap import core, real
    >>> bounds = [(0, 1), (0, 1), (-1, 100)]
    >>> init = initialize_vectors_uniform(bounds)
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
        return [list(generate_genome()) for _ in range(pop_size)]

    return f


##############################
# Class Spheroid
##############################
class Spheroid(ScalarProblem):
    """ Classic parabolid function, known as the "sphere" or "spheroid" problem, because its equal-fitness contours form (hyper)spheres in n > 2.

    .. math::
    
       f(\\vec{x}) = \sum_{i}^n x_i^2

    .. plot::
       :include-source:
   
       from leap import real
       bounds = real.Spheroid.bounds  # Contains traditional bounds
       real.plot_2d_problem(real.Spheroid(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds for a sphere functions solution."""
    bounds = (-5.12, 5.12)

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

        >>> s = Spheroid()
        >>> s.worse_than(100, 10)
        False

        >>> s = Spheroid(maximize=False)
        >>> s.worse_than(100, 10)
        True
        """
        return super().worse_than(first_fitness, second_fitness)


##############################
# Class Rastrigin
##############################
class Rastrigin(ScalarProblem):
    """ The classic Rastrigin problem.  The Rastrigin provides a real-valued fitness landscape with a quadratic global structure (like the :class:`~leap.real.Spheroid`), plus a sinusoidal local structure with many local optima.

    .. math::
    
       f(\\vec{x}) = An + \sum_{i=1}^n x_i^2 - A\cos(2\pi x_i)

    .. plot::
       :include-source:
   
       from leap import real
       bounds = real.Rastrigin.bounds  # Contains traditional bounds
       real.plot_2d_problem(real.Rastrigin(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds."""
    bounds = (-5.12, 5.12)

    # TODO See if we get an error if we try to add a constructor that doesn't set maximize

    def __init__(self, a=1.0, maximize=True):
        super().__init__(maximize)
        self.a = a

    def evaluate(self, phenome):
        """
        Computes the spheroid function from a real-valued list phenome:

        >>> phenome = [1.0/12, 0]
        >>> Rastrigin().evaluate(phenome) # +doctest: ELLIPSIS
        3.872969...

        :param phenome: to be evaluated
        """
        return self.a*len(phenome) + sum([x**2 + self.a*np.cos(2*np.pi*x) for x in phenome])

    def worse_than(self, first_fitness, second_fitness):
        """
        We maximize by default:

        >>> s = Rastrigin()
        >>> s.worse_than(100, 10)
        False

        >>> s = Rastrigin(maximize=False)
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


##############################
# Function plot_2d_problem
##############################
def plot_2d_problem(problem, xlim, ylim, ax=None, granularity=0.1):
    """
    Convenience function for plotting a :class:`~leap.problem.Problem` that accepts 2-D real-valued phenomes and produces a 1-D scalar fitness output.

    :param ~leap.problem.Problem fun: The :class:`~leap.problem.Problem` to plot.
    :param xlim: Bounds of the horizontal axes.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will be created).
    :param float granularity: Spacing of the grid to sample points along.


    The difference between this and :meth:`plot_2d_function` is that this takes a :class:`~leap.problem.Problem` object (instead of a raw function).

    If no axes are specified, a new figure is created for the plot:

    .. plot::
       :include-source:
   
       from leap import real
       problem = real.CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])
       real.plot_2d_problem(problem, xlim=(0, 1), ylim=(0, 1), granularity=0.025);

    You can also specify axes explicitly (ex. by using `ax=plt.gca()`.  You  must configure your axes to use `projection='3d'`:

    .. plot::
       :include-source:

       from matplotlib import pyplot as plt
       from leap import real
       fig = plt.figure(figsize=(12, 4))
       plt.subplot(121, projection='3d')
       real.plot_2d_problem(real.Spheroid(), ax=plt.gca(), xlim=(-5.12, 5.12), ylim=(-5.12, 5.12))
       plt.subplot(122, projection='3d')
       real.plot_2d_problem(real.Rastrigin(), ax=plt.gca(), xlim=(-5.12, 5.12), ylim=(-5.12, 5.12))
       
    """
    def call(phenome):
        return problem.evaluate(phenome)
    return plot_2d_function(call, xlim, ylim, ax, granularity)


##############################
# Function plot_2d_function
##############################
def plot_2d_function(fun, xlim, ylim, ax=None, granularity=0.1):
    """
    Convenience method for plotting a function that accepts 2-D real-valued imputs and produces a 1-D scalar output.

    :param function fun: The function to plot.
    :param xlim: Bounds of the horizontal axes.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will be created).
    :param float granularity: Spacing of the grid to sample points along.

    The difference between this and :meth:`plot_2d_problem` is that this takes a raw function (instead of a :class:`~leap.problem.Problem` object).

    .. plot::
       :include-source:

       import numpy as np
       from scipy import linalg

       from leap import real

       def sinc_hd(phenome):
           r = linalg.norm(phenome)
           return np.sin(r)/r

       real.plot_2d_function(sinc_hd, xlim=(-10, 10), ylim=(-10, 10), granularity=0.2)


    """
    assert(len(xlim) == 2)
    assert(len(ylim) == 2)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    @np.vectorize
    def v_fun(x, y):
        return fun([x, y])

    x = np.arange(xlim[0], xlim[1], granularity)
    y = np.arange(ylim[0], ylim[1], granularity)
    xx, yy = np.meshgrid(x, y)

    return ax.plot_surface(xx, yy, v_fun(xx, yy))
