#!/usr/bin/env python3
"""
    LEAP Problem classes for multiobjective optimization.
"""
from collections import Sequence

import numpy as np

from ..problem import Problem

##############################
# Class MultiObjectiveProblem
##############################
class MultiObjectiveProblem(Problem):
    """A problem that compares individuals based on Pareto dominance.

    Inherit from this class and implement the `evaluate()` method to implement
    an objective function that returns a list of real-value fitness values.

    In Pareto-dominance, an individual A is only considered "better than" an individual
    B if A is unambiguously better than B: i.e. it is at least as good as B on
    all objectives, and it is strictly better than B on at least one objective.

    .. plot::

        from matplotlib import pyplot as plt
        plt.rcParams.update({ "text.usetex": True })

        plt.figure(figsize=(8, 6))
        plt.plot([1.0], [1.0], marker='o', markersize=10, color='black')
        plt.annotate("$A$", (1.04, 0.9), fontsize='x-large')
        plt.axvline(1.0, linestyle='dashed', color='black')
        plt.axhline(1.0, linestyle='dashed', color='black')
        plt.annotate("Dominates A", (1.3, 1.5), fontsize='xx-large')
        plt.annotate("$\\succ A$", (1.45, 1.35), fontsize='xx-large')
        plt.annotate("$\\prec A$", (0.45, 0.35), fontsize='xx-large')
        plt.annotate("Neither dominated\\nnor dominating", (0.25, 1.4), fontsize='xx-large')
        plt.annotate("Neither dominated\\nnor dominating", (1.25, 0.4), fontsize='xx-large')
        plt.annotate("Dominated by A", (0.25, 0.5), fontsize='xx-large')
        plt.axvspan(0, 1.0, ymin=0, ymax=0.5, alpha=0.5, color='red')
        plt.axvspan(1.0, 2.0, ymin=0.5, ymax=1.0, alpha=0.5, color='blue')
        plt.axvspan(1.0, 2.0, ymin=0, ymax=0.5, alpha=0.1, color='gray')
        plt.axvspan(0, 1.0, ymin=0.5, ymax=1.0, alpha=0.1, color='gray')
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.xlabel("Objective 1", fontsize=15)
        plt.ylabel("Objective 2", fontsize=15)
        plt.title("Pareto dominance in two dimensions", fontsize=20)

    """
    def __init__(self, maximize: list):
        """
        :param maximize: a list of booleans where True indicates a given feature
            is a maximization objective, else minimization.
        """
        super().__init__()

        assert(maximize is not None)
        assert(len(maximize) > 0)
        # Represent maximize as a vector of 1's and -1's; this is used in
        # worse_than() to ensure we are always dealing with maximization by
        # converting objectives to maximization objectives as needed.
        # E.g., for l = [True, False, True, True]
        #   1 * np.array(l) - 1 * np.invert(l) -> array([ 1, -1,  1,  1])
        self.maximize = 1 * np.array(maximize) - 1 * np.invert(maximize)

    def worse_than(self, first_fitnesses, second_fitnesses):
        """Return true if first_fitnesses is Pareto-dominated by second_fitnesses.

        In the case of maximization over all objectives, a solution :math:`b`
        dominates :math:`a`, written :math:`b \succ a`, if and only if

        .. math::

              \\begin{array}{ll}
                f_i(b) \\ge f_i(a) & \\forall i, \\text{ and} \\\\
                f_i(b) > f_j(a) & \\text{ for some } j.
              \\end{array}

        Here we may maximize over some objectives, and minimize over others,
        depending on the values in the `self.maximize` list.

        :param first_fitnesses: a np array of real-valued fitnesses for an
            individual, where each element corresponds to a single objective
        :param second_fitnesses: same as `first_fitnesses`, but for a different
            individual
        """
        assert(first_fitnesses is not None)
        assert(second_fitnesses is not None)
        assert(len(first_fitnesses) == len(self.maximize))
        assert(len(second_fitnesses) == len(self.maximize))

        # Negate the minimization problems, so we can treat all objectives as
        # maximization
        first_max = first_fitnesses * self.maximize
        second_max = second_fitnesses * self.maximize

        # Now check the two conditions for dominance using numpy comparisons
        return all (second_max >= first_max) \
                and any (second_max > first_max)

    def equivalent(self, first_fitnesses, second_fitnesses):
        """Return true if first_fitness and second_fitness are mutually
        Pareto non-dominating.

        .. math::
            a \\not \\succ b \\text{ and } b \\not \\succ a

        :param first_fitnesses: a np array of real-valued fitnesses for an
            individual, where each element corresponds to a single objective
        :param second_fitnesses: same as `first_fitnesses`, but for a different
            individual
        """
        return not self.worse_than(first_fitnesses, second_fitnesses) \
               and not self.worse_than(second_fitnesses, first_fitnesses)



##############################
# Class ZDTBenchmarkProblem
##############################
class ZDTBenchmarkProblem(MultiObjectiveProblem):
    """ A problem that implements Kalyanmoy Deb's popular tunable two-objective
    problem 'toolkit.'

    This allows us to create custom two-objective functions by defining three functions:
    the first objective :math:`f_1(y)`, a second function :math:`g(x)`, and an extra
    function :math:`h(f_1, g)` that governs how the functions interact to produce
    the second objective :math:`f_2(x)`:

    .. math::

        \\begin{array}{ll}
        \\text{Given} & \\mathbf{x} = \\{ x_1, \\dots, x_n \\} \\\\
        \\text{Minimize} & (f_1(\\mathbf{y}), f_2(\\mathbf{y}, \\mathbf{z})) \\\\
        \\text{where} & \\begin{aligned}[t]
            f_2(\\mathbf{y}, \\mathbf{z}) &= g(\\mathbf{z}) \\times h(f_1(\\mathbf{y}), g(\\mathbf{z})) \\\\
            \\mathbf{y} &= \\{ x_1, \dots, x_j \\} \\\\
            \\mathbf{z} &= \\{ x_{j+1}, \dots, x_n \\}
            \end{aligned}
        \\end{array}

    This framework is used to define several classic multi-objective benchmark problems,
    such as :py:class:`leap_ec.real_rep.problems.ZDT1Problem`, etc.

    - Deb, Kalyanmoy. "Multi-objective genetic algorithms: Problem difficulties and
      construction of test problems." *Evolutionary computation* 7.3 (1999): 205-230.
    """
    def __init__(self, f, n: int, g = None, maximize: list = None):
        """
        :param f: list of two or more objective functions
        :param n: problem dimension
        :param g: optional function that describes interactions between
            functions defined in `f`
        :param maximize: list of booleans that map to the functions in `f`, and
            are True if the corresponding function is maximizing
        """
        assert isinstance(f, Sequence)
        if maximize is not None:
            # Ya'd better have specified min/max by objectives for the same
            # number of objective functions.
            assert len(maximize) == len(f)
        else:
            # by default we assume minimizing for all objectives
            maximize = False * len(f)
        assert n > 0
        # assert(g is not None) Not all Deb benchmark functions have a g()
        if g is not None:
            assert(callable(g))
        super().__init__(maximize)
        self.f = f
        self.n = n
        self.g = g


    # def evaluate(self, phenome, *args, **kwargs):
    #
    #     y = phenome[:self.n]
    #     z = phenome[self.n:]
    #
    #     o1 = self.f1(y)
    #     g_out = self.g(z)
    #     o2 = g_out * h(o1, g_out)
    #     return (o1, o2)


##############################
# Class SCHProblem
##############################
class SCHProblem(MultiObjectiveProblem):
    """ SCH problem from Deb et al's benchmarks

    .. math::

        \\begin{align}
        f_1(x) &= x^2 \\\\
        f_2(x) &= (x-2)^2 \\\\
        -10^3 \\le x &\\le 10^3
        \\end{align}

    - Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
      "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
      transactions on evolutionary computation 6, no. 2 (2002): 182-197.
    """
    def __init__(self):
        super.__init__(maximize=(False,False))

    def evaluate(self, phenome, *args, **kwargs):
        """
        :param phenome: argument for objective functions
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        fitness = np.zeros(2)
        fitness[0] = phenome[0] ** 2
        fitness[1] = (phenome[0] - 2) ** 2
        return fitness


##############################
# Class ZTD1Problem
##############################
class ZDT1Problem(ZDTBenchmarkProblem):
    """
    The first problem from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite.  It's meant to provide a simple multi-objective problem with a *convex*
    Pareto-optimal front.

    This function is defined via the :py:class:`leap_ec.problem.ZDTBenchmarkProblem`
    with the following parameters:

    .. math::

        \\begin{align}
        f_1(x) &= x_1 \\\\
        f_2(x) &= g(x)[1-\sqrt{x_1/g(x)}] \\\\
        g(x) &= 1 + 9\\frac{\sum_{i=2}^n x_i}{n - 1} \\\\
        0 \\le x_i &\\le 1, \mbox{ } i = 1, \dots, n
        \\end{align}

    Traditionally the problem is used with :math:`|x| = 30` dimensions in the solution space.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    def __init__(self, n = 30, maximize = None):
        """
        :param n: number of dimensions
        :param maximize: boolean vector for each objective where True means
            corresponding objective function is maximized
        """
        if maximize is None:
            maximize = (False,) * n
        super().__init__(maximize=maximize)
        self.n = n

    def evaluate(self, phenome, *args, **kwargs):
        """
        :param phenome: contains x
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        def g(x):
            """:param x: phenome[1..n]"""
            return 1.0 + 9.0 * (x.sum()) / (self.n - 1.0)
        fitness = np.zeros(2)
        fitness[0] = phenome[0]
        fitness[1] = g(phenome[1:]) * (1 - np.sqrt(phenome[0]/g(phenome[1:])))

        return fitness



##############################
# Class ZTD2Problem
##############################
class ZDT2Problem(ZDTBenchmarkProblem):
    """
    The second problem from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite.  This is similar to :py:class:`leap_ec.problem.ZDT1Problem`, except that
    it has a *non-convex* Pareto front.

    This function is defined via the :py:class:`leap_ec.problem.ZDTBenchmarkProblem`
    with the following parameters:

    .. math::

        \\begin{align}
        f_1(x) &= x_1 \\\\
        f_2(x) &= g(x)[1-(x_1/g(x))^2] \\\\
        g(x) &= 1 + 9\\frac{\sum_{i=2}^n x_i}{n - 1} \\\\
        0 \\le x_i &\\le 1, \mbox{ } i = 1, \dots, n
        \\end{align}

    Traditionally the problem is used with :math:`|x| = 30` dimensions in the solution space.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    def __init__(self, n = 30, maximize = None):
        """
         :param n: number of dimensions
         :param maximize: boolean vector for each objective where True means
             corresponding objective function is maximized
         """
        if maximize is None:
            maximize = (False,) * n
        super().__init__(maximize=maximize)
        self.n = n


    def evaluate(self, phenome, *args, **kwargs):
        """
        :param phenome: contains x
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        def g(x):
            """:param x: phenome[1..n]"""
            return 1.0 + 9.0 * (x.sum()) / (self.n - 1.0)
        fitness = np.zeros(2)
        fitness[0] = phenome[0]
        fitness[1] = g(phenome[1:]) * (1 - (phenome[0] / g(phenome[1:])) ** 2)

        return fitness
