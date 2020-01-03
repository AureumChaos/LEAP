"""Functions and objects for working with real-valued optimization problems."""

import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from leap.problem import ScalarProblem

from leap import core



##############################
# Class Spheroid
##############################
class Spheroid(ScalarProblem):
    """ Classic parabolid function, known as the "sphere" or "spheroid" problem, because its equal-fitness contours form (hyper)spheres in n > 2.

    .. math::

       f(\\vec{x}) = \\sum_{i}^n x_i^2

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
        Computes the function value from a real-valued list phenome:

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

       f(\\vec{x}) = An + \\sum_{i=1}^n x_i^2 - A\\cos(2\\pi x_i)

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
        Computes the function value from a real-valued list phenome:

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
# Class Rosenbrock
##############################
class Rosenbrock(ScalarProblem):
    """ The classic Rosenbrock problem, a.k.a. the "banana" or "valley" function.

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^{d-1} \\left[ 100 (x_{i + 1} - x_i^2)^2 + (x_i - 1)^2\\right]

    .. plot::
       :include-source:

       from leap import real
       bounds = real.Rosenbrock.bounds  # Contains traditional bounds
       real.plot_2d_problem(real.Rosenbrock(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds."""
    bounds = (-2.048, 2.048)

    # TODO See if we get an error if we try to add a constructor that doesn't set maximize

    def __init__(self, maximize=True):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome:

        >>> phenome = [0.5, -0.2, 0.1]
        >>> Rosenbrock().evaluate(phenome)
        22.3

        :param phenome: to be evaluated
        """
        sum = 0
        # TODO Speed this up with numpy
        for i, x in enumerate(phenome[0:-1]):
            x_p = phenome[i + 1]
            sum += 100*(x_p - x**2)**2 + (x - 1)**2
        return sum

    def worse_than(self, first_fitness, second_fitness):
        """
        We maximize by default:

        >>> s = Rosenbrock()
        >>> s.worse_than(100, 10)
        False

        >>> s = Rosenbrock(maximize=False)
        >>> s.worse_than(100, 10)
        True
        """
        return super().worse_than(first_fitness, second_fitness)


##############################
# Class StepProblem
##############################
class StepProblem(ScalarProblem):
    """ The classic 'step' function—a function with a linear global structure, but with stair-like plateaus at the local
    level.

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^{n} \\lfloor x_i \\rfloor

    where :math:`\\lfloor x \\rfloor` denotes the floor function.

    .. plot::
       :include-source:

       from leap import real
       bounds = real.StepProblem.bounds  # Contains traditional bounds
       real.plot_2d_problem(real.StepProblem(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds."""
    bounds = (-5.12, 5.12)

    # TODO See if we get an error if we try to add a constructor that doesn't set maximize

    def __init__(self, maximize=True):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome:

        >>> phenome = [3.5, -3.8, 5.0]
        >>> StepProblem().evaluate(phenome)
        4.0

        :param phenome: to be evaluated
        """
        return np.sum(np.floor(phenome))

    def worse_than(self, first_fitness, second_fitness):
        """
        We maximize by default:

        >>> s = StepProblem()
        >>> s.worse_than(100, 10)
        False

        >>> s = StepProblem(maximize=False)
        >>> s.worse_than(100, 10)
        True
        """
        return super().worse_than(first_fitness, second_fitness)


##############################
# Class NoisyQuartic
##############################
class NoisyQuartic(ScalarProblem):
    """ The classic 'quadratic quartic' function with Gaussian noise:

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^{n} i x_i^4 + \\texttt{gauss}(0, 1)

    .. plot::
       :include-source:

       from leap import real
       bounds = real.NoisyQuartic.bounds  # Contains traditional bounds
       real.plot_2d_problem(real.NoisyQuartic(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds."""
    bounds = (-1.28, 1.28)

    # TODO See if we get an error if we try to add a constructor that doesn't set maximize

    def __init__(self, maximize=True):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome (the output varies, since the function has noise):

        >>> phenome = [3.5, -3.8, 5.0]
        >>> r = NoisyQuartic().evaluate(phenome)
        >>> print(f'Result: {r}')
        Result: ...

        :param phenome: to be evaluated
        """
        indices = np.arange(len(phenome))
        noise = np.random.normal(0, 1, len(phenome))
        return np.sum(np.dot(indices, np.power(phenome, 4)) + noise)

    def worse_than(self, first_fitness, second_fitness):
        """
        We maximize by default:

        >>> s = NoisyQuartic()
        >>> s.worse_than(100, 10)
        False

        >>> s = NoisyQuartic(maximize=False)
        >>> s.worse_than(100, 10)
        True
        """
        return super().worse_than(first_fitness, second_fitness)


##############################
# Class ShekelProblem
##############################
class ShekelProblem(ScalarProblem):
    """ The classic 'Shekel's foxholes' function.

    .. math::

       f(\\mathbf{x}) = \\frac{1}{\\frac{1}{K} + \\sum_{j=1}^{25} \\frac{1}{f_j(\\mathbf{x})}}


    where

    .. math::

       f_j(\\mathbf{x}) = c_j + \\sum_{i=1}^2 (x_i - a_{ij})^6

    and the points :math:`\\left\\{ (a_{1j}, a_{2j})\\right\\}_{j=1}^{25}` define the functions various optima, and are
    given by the following hardcoded matrix:

    .. math::

       \\left[a_{ij}\\right] = \\left[ \\begin{array}{lllllllllll}
                                        -32 & -16 & 0 & 16 & 32 & -32 & -16 & \\cdots & 0 & 16 & 32 \\\\
                                        -32 & -32 & -32 & -32 & -32 & -16 & -16 & \\cdots & 32 & 32 & 32
                                       \\end{array} \\right].

    :param int k: the value of :math:`K` in the fitness function.
    :param [int] c: list of values for the function's :math:`c_j` parameters.  Each `c[j]` approximately corresponds to
        the depth of the jth foxhole.
    :param maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap import real
       bounds = real.ShekelProblem.bounds  # Contains traditional bounds
       real.plot_2d_problem(real.ShekelProblem(), xlim=bounds, ylim=bounds, granularity=0.9)

    """

    """ Standard bounds."""
    bounds = (-65.536, 65.536)

    points = np.array([[-32, -16, 0, 16, 32]*5,
                       [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5])

    # TODO See if we get an error if we try to add a constructor that doesn't set maximize

    def __init__(self, k=500, c=np.arange(1, 26), maximize=True):
        super().__init__(maximize)
        self.k = k
        self.c = c

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome (the output varies, since the function has noise).

        :param phenome: to be evaluated
        """
        assert(len(phenome) == 2)

        def f(j):
            return self.c[j] + (phenome[0] - self.points[0][j])**6 + (phenome[1] - self.points[1][j])**6

        return 1/(1/self.k + np.sum([1/f(j) for j in range(25)]))

    def worse_than(self, first_fitness, second_fitness):
        """
        We maximize by default:

        >>> s = ShekelProblem()
        >>> s.worse_than(100, 10)
        False

        >>> s = ShekelProblem(maximize=False)
        >>> s.worse_than(100, 10)
        True
        """
        return super().worse_than(first_fitness, second_fitness)


##############################
# Class CosineFamilyProblem
##############################
class CosineFamilyProblem(ScalarProblem):
    """
    A configurable multi-modal function based on combinations of cosines, taken from the problem generators proposed
    in

     * Jani Rönkkönen et al., "A Generator for Multimodal Test Functions with Multiple Global Optima," *Asia-Pacific Conference on Simulated Evolution and Learning*. Springer, Berlin, Heidelberg, 2008.

    .. math::

       f_{\\cos}(\\mathbf{x}) = \\frac{\\sum_{i=1}^n -\\cos((G_i - 1)2 \\pi x_i)
                                - \\alpha \\cdot \\cos((G_i - 1)2 \\pi L-i x_y)}{2n}

    where :math:`G_i` and :math:`L_i` are parameters that indicate the number of global and local optima, respectively,
    in the ith dimension.

    :param float alpha: parameter that controls the depth of the local optima.
    :param [int] global_optima_counts: list of integers indicating the number of global optima for each dimension.
    :param [int] local_optima_counts: list of integers indicated the number of local optima for each dimension.
    :param maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap import real
       problem = real.CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])
       bounds = real.CosineFamilyProblem.bounds  # Contains traditional bounds
       real.plot_2d_problem(problem, xlim=bounds, ylim=bounds, granularity=0.025)

    The number of optima can be varied independently by each dimension:

    .. plot::
       :include-source:

       from leap import real
       problem = real.CosineFamilyProblem(alpha=3.0, global_optima_counts=[4, 2], local_optima_counts=[2, 2])
       bounds = real.CosineFamilyProblem.bounds  # Contains traditional bounds
       real.plot_2d_problem(problem, xlim=bounds, ylim=bounds, granularity=0.025)

    """

    bounds = (0, 1)

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
# Class TranslatedProblem
##############################
class TranslatedProblem(ScalarProblem):
    """
    Takes an existing fitness function and translates it by applying a fixed offset vector.

    For example,

    .. plot::
       :include-source:

       from leap import real

       original_problem = real.Spheroid()
       offset = [-1.0, -2.5]
       translated_problem = real.TranslatedProblem(original_problem, offset)

       fig = plt.figure(figsize=(12, 8))

       plt.subplot(221, projection='3d')
       bounds = real.Spheroid.bounds  # Contains traditional bounds
       real.plot_2d_problem(original_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(222, projection='3d')
       real.plot_2d_problem(translated_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(223)
       real.plot_2d_problem(original_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(224)
       real.plot_2d_problem(translated_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)
    """
    def __init__(self, problem, offset, maximize=True):
        super().__init__(maximize=maximize)
        assert(problem is not None)
        self.problem = problem
        self.offset = np.array(offset)
        if hasattr(problem, 'bounds'):
            self.bounds = problem.bounds

    @classmethod
    def random(cls, problem, offset_bounds, dimensions, maximize=True):
        """ Apply a random real-valued translation to a fitness function, sampled uniformly between min_offset and
        max_offset in every dimension.

        .. plot::
           :include-source:

           from leap import real

           original_problem = real.Rastrigin()
           bounds = real.Rastrigin.bounds  # Contains traditional bounds
           translated_problem = real.TranslatedProblem.random(original_problem, bounds, 2)

           real.plot_2d_problem(translated_problem, kind='contour', xlim=bounds, ylim=bounds)
        """
        min_offset, max_offset = offset_bounds
        offset = np.random.uniform(min_offset, max_offset, dimensions)
        return cls(problem, offset, maximize=maximize)

    def evaluate(self, phenome):
        """
        Evaluate the fitness of a point after translating the fitness function.

        Translation can be used in higher than two dimensions:

        >>> offset = [-1.0, -1.0, 1.0, 1.0, -5.0]
        >>> t_sphere = TranslatedProblem(Spheroid(), offset)
        >>> genome = [0.5, 2.0, 3.0, 8.5, -0.6]
        >>> t_sphere.evaluate(genome)
        90.86
        """
        assert (len(phenome) == len(self.offset)), \
            f"Tried to evalute a {len(phenome)}-D genome in a {len(self.offset)}-D fitness function."
        # Substract the offset so that we are moving the origin *to* the offset.
        # This way we can think of it as offsetting the fitness function, rather than the input points.
        new_phenome = np.array(phenome) - self.offset
        return self.problem.evaluate(new_phenome)


################################
# Class MatrixTransformedProblem
################################
class MatrixTransformedProblem(ScalarProblem):
    """ Apply a linear transformation to a fitness function.

    :param matrix: an nxn matrix, where n is the genome length.
    :returns: a function that first applies -matrix to the input, then applies fun to the transformed input.

    For example, here we manually construct a 2x2 rotation matrix and apply it to the :class:`~leap.real.Rosenbrock`
    function:

    .. plot::
       :include-source:

       from leap import real

       original_problem = real.Rosenbrock()
       theta = np.pi/2
       matrix = [[np.cos(theta), -np.sin(theta)],\
                 [np.sin(theta), np.cos(theta)]]

       transformed_problem = real.MatrixTransformedProblem(original_problem, matrix)

       fig = plt.figure(figsize=(12, 8))

       plt.subplot(221, projection='3d')
       bounds = real.Rosenbrock.bounds  # Contains traditional bounds
       real.plot_2d_problem(original_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(222, projection='3d')
       real.plot_2d_problem(transformed_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(223)
       real.plot_2d_problem(original_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(224)
       real.plot_2d_problem(transformed_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

    """
    def __init__(self, problem, matrix, maximize=True):
        super().__init__(maximize=maximize)
        assert(problem is not None)
        assert(len(matrix) == len(matrix[0]))
        self.matrix = np.array(matrix)
        self.problem = problem
        if hasattr(problem, 'bounds'):
            self.bounds = problem.bounds

    @classmethod
    def random_orthonormal(cls, problem, dimensions, maximize=True):
        """Create a :class:`~leap.real.MatrixTransformedProblem` that performs a random rotation and/or inversion of the
        function.

        We accomplish this by generating a random orthonormal basis for R^n and plugging the resulting matrix into
        :class:`~leap.real.MatrixTransformedProblem`.

        This algorithm follows directly from the definition of orthonormality.  It is described in Hansen and
        Ostermeier's original CMA-ES paper: "Completely derandomized self-adaptation in evolution strategies."
        Evolutionary computation 9.2 (2001): 159-195.

        :param problem: the original :class:`~leap.real.ScalarProblem` to apply the transform to.
        :param dimensions: the number of elements each vector should have.

        .. plot::
           :include-source:

           from leap import real

           original_problem = real.CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 3], local_optima_counts=[2, 3])

           transformed_problem = real.MatrixTransformedProblem.random_orthonormal(original_problem, 2)

           fig = plt.figure(figsize=(12, 8))

           plt.subplot(221, projection='3d')
           bounds = original_problem.bounds
           real.plot_2d_problem(original_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

           plt.subplot(222, projection='3d')
           real.plot_2d_problem(transformed_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

           plt.subplot(223)
           real.plot_2d_problem(original_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

           plt.subplot(224)
           real.plot_2d_problem(transformed_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)
        """
        matrix = np.random.normal(size=[dimensions, dimensions])
        for i, row in enumerate(matrix):
            previous_rows = matrix[0:i, :]
            matrix[i, :] = row - sum([np.dot(row, prev) * prev for prev in previous_rows])
            matrix[i, :] = row / np.linalg.norm(row)

        # Any vector in the resulting matrix will be of unit length
        assert(round(np.linalg.norm(matrix[0], 5)) == 1.0)
        # Any pair of vectors will be linearly independent
        assert(abs(round(np.dot(matrix[0], matrix[1]), 5)) == 0.0)

        return cls(problem, matrix, maximize)

    def evaluate(self, phenome):
        """
        Evaluated the fitness of a point on the transformed fitness landscape.

        For example, consider a sphere function whose global optimum is situated at (0, 1):

        >>> from leap import real_problems
        >>> s = real_problems.TranslatedProblem(real_problems.Spheroid(), offset=[0, 1])
        >>> round(s.evaluate([0, 1]), 5)
        0

        Now let's take a rotation matrix that transforms the space by pi/2 radians:

        >>> import numpy as np
        >>> theta = np.pi/2
        >>> matrix = [[np.cos(theta), -np.sin(theta)],\
                      [np.sin(theta), np.cos(theta)]]
        >>> r = MatrixTransformedProblem(s, matrix)

        The rotation has moved the new global optimum to (1, 0)

        >>> round(r.evaluate([1, 0]), 5)
        0.0

        The point (0, 1) lies at a distance of sqrt(2) from the new optimum, and has a fitness of 2:

        >>> round(r.evaluate([0, 1]), 5)
        2.0
        """
        assert(len(phenome) == len(self.matrix)), f"Tried to evalute a {len(phenome)}-D genome in a {len(self.matrix)}-D fitness function."
        new_point = np.matmul(self.matrix, phenome)
        return self.problem.evaluate(new_point)


##############################
# Function plot_2d_problem
##############################
def plot_2d_problem(problem, xlim, ylim, kind='surface', ax=None, granularity=0.1):
    """
    Convenience function for plotting a :class:`~leap.problem.Problem` that accepts 2-D real-valued phenomes and produces a 1-D scalar fitness output.

    :param ~leap.problem.Problem fun: The :class:`~leap.problem.Problem` to plot.
    :param xlim: Bounds of the horizontal axes.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param kind: The kind of plot to create: 'surface' or 'contour'
    :type kind: str
    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will be created).
    :param float granularity: Spacing of the grid to sample points along.


    The difference between this and :meth:`plot_2d_function` is that this takes a :class:`~leap.problem.Problem` object (instead of a raw function).

    If no axes are specified, a new figure is created for the plot:

    .. plot::
       :include-source:

       from leap import real
       problem = real.CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])
       real.plot_2d_problem(problem, xlim=(0, 1), ylim=(0, 1), granularity=0.025);

    You can also specify axes explicitly (ex. by using `ax=plt.gca()`.  When plotting surfaces, you  must configure your
    axes to use `projection='3d'`.  Contour plots don't need 3D axes:

    .. plot::
       :include-source:

       from matplotlib import pyplot as plt
       from leap import real

       fig = plt.figure(figsize=(12, 4))
       bounds=real.Rastrigin.bounds  # Contains default bounds

       plt.subplot(121, projection='3d')
       real.plot_2d_problem(real.Rastrigin(), ax=plt.gca(), xlim=bounds, ylim=bounds)

       plt.subplot(122)
       real.plot_2d_problem(real.Rastrigin(), ax=plt.gca(), kind='contour', xlim=bounds, ylim=bounds)

    """
    def call(phenome):
        return problem.evaluate(phenome)

    if kind == 'surface':
        return plot_2d_function(call, xlim, ylim, ax, granularity)
    elif kind == 'contour':
        return plot_2d_contour(call, xlim, ylim, ax, granularity)
    else:
        raise ValueError(f'Unrecognized plot kind: "{kind}".')


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


##############################
# Function plot_2d_contour
##############################
def plot_2d_contour(fun, xlim, ylim, ax=None, granularity=0.1):
    """
    Convenience method for plotting contours for a function that accepts 2-D real-valued inputs and produces a 1-D
    scalar output.

    :param function fun: The function to plot.
    :param xlim: Bounds of the horizontal axes.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will be created).
    :param float granularity: Spacing of the grid to sample points along.

    The difference between this and :meth:`plot_2d_problem` is that this takes a raw function (instead of a
    :class:`~leap.problem.Problem` object).

    .. plot::
       :include-source:

       import numpy as np
       from scipy import linalg

       from leap import real

       def sinc_hd(phenome):
           r = linalg.norm(phenome)
           return np.sin(r)/r

       real.plot_2d_contour(sinc_hd, xlim=(-10, 10), ylim=(-10, 10), granularity=0.2)


    """
    assert(len(xlim) == 2)
    assert(len(ylim) == 2)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    @np.vectorize
    def v_fun(x, y):
        return fun([x, y])

    x = np.arange(xlim[0], xlim[1], granularity)
    y = np.arange(ylim[0], ylim[1], granularity)
    xx, yy = np.meshgrid(x, y)

    return ax.contour(xx, yy, v_fun(xx, yy))
