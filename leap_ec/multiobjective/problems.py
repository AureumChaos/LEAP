#!/usr/bin/env python3
"""
    LEAP Problem classes for multiobjective optimization.
"""
from typing import Sequence
from abc import abstractmethod
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
    def __init__(self, maximize: Sequence[bool]):
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
        # Fix for maximize bug courtesy of Luke McCombs; he suggested using np.where()
        self.maximize = np.where(maximize, 1, -1)

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
        # assert(first_fitnesses is not None)
        # assert(second_fitnesses is not None)
        # assert(len(first_fitnesses) == len(self.maximize))
        # assert(len(second_fitnesses) == len(self.maximize))

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
# Class SCHProblem
##############################
class SCHProblem(MultiObjectiveProblem):
    """ SCH problem from Deb et al's benchmarks

    This expects a numpy scalar (zero dimensional) for a phenome.

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
        super().__init__(maximize=(False, False))

    def evaluate(self, phenome):
        """
        :param phenome: argument for objective functions
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        fitness = np.zeros(2)
        fitness[0] = phenome ** 2
        fitness[1] = (phenome - 2.0) ** 2
        return fitness

class ZDTBenchmarkProblem(MultiObjectiveProblem):
    """
    The base class for problems from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite.
    
    Each problem is of the form:
    
    .. math::

        \\begin{align}
        \\textrm{Minimize} \\quad& \\mathcal{T}(x) &=&\\quad (f_1(x_1), f_2(x)) \\\\
        \\textrm{subject to} \\quad& f_2(x) &=&\\quad g(x_2, \\ldots, x_n)h(f_1(x_1), g(x_2, \\ldots, x_n))\\\\
        \\textrm{where} \\quad& x &=&\\quad (x_1,\\ldots,x_m)\\\\
        \\end{align}
        
    For reliability when testing, each problem has been provided with a `check_phenome`
    parameter to ensure that phenomes match the expected form and bounds of the problem.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    
    def __init__(self, n, check_phenome=True):
        """
        :param n: number of dimensions
        :param check_phenome: whether phenome bounds checking should be performed
        """
        super().__init__(maximize=(False, False))
        self.n = n
        self.check_phenome = check_phenome
    
    @property
    @abstractmethod
    def bounds(self):
        """
        :returns: the bounds of the phenome
        """
        pass

##############################
# Class ZDT1Problem
##############################
class ZDT1Problem(ZDTBenchmarkProblem):
    """
    The first problem from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite.  It's meant to provide a simple multi-objective problem with a *convex*
    Pareto-optimal front.
    
    .. math::

        \\begin{align}
        f_1(x_1) &= x_1 \\\\
        g(x_2, \\ldots, x_n) &= 1 + 9\\cdot {\\sum_{i=2}^n x_i} / (n - 1) \\\\
        h(f_1, g) &= 1 - \\sqrt{f_1/g}\\\\
        f_2(x) &= g(x_2, \\ldots, x_n)h(f_1(x_1), g(x_2, \\ldots, x_n))\\\\
        \\\\\\\\
        x_i &\\in [0,1]
        \\end{align}

    Traditionally the problem is used with :math:`|x| = 30` dimensions in the solution space.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    def __init__(self, n=30, check_phenome=True):
        super().__init__(n, check_phenome)
    
    @property
    def bounds(self):
        return [(0, 1)] * self.n

    def evaluate(self, phenome):
        """
        :param phenome: contains x
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        if self.check_phenome:
            assert ((phenome >= 0) & (phenome <= 1)).all(),\
                f"All elements of phenome must be in [0.0, 1.0]: {phenome}"
        
        def g(x):
            """:param x: phenome[1..n]"""
            return 1 + 9 * (x.sum()) / (self.n - 1)
        
        fitness = np.zeros(2)
        fitness[0] = phenome[0]
        
        g_x = g(phenome[1:]) 
        h = 1 - np.sqrt(phenome[0] / g_x)
            
        fitness[1] = g_x * h

        return fitness



##############################
# Class ZDT2Problem
##############################
class ZDT2Problem(ZDTBenchmarkProblem):
    """
    The second problem from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite.  This is similar to :py:class:`leap_ec.problem.ZDT1Problem`, except that
    it has a *non-convex* Pareto front.

    .. math::

        \\begin{align}
        f_1(x_1) &= x_1 \\\\
        g(x_2, \\ldots, x_n) &= 1 + 9\\cdot {\\sum_{i=2}^n x_i} / (n - 1) \\\\
        h(f_1, g) &= 1 - (f_1/g)^2\\\\
        f_2(x) &= g(x_2, \\ldots, x_n)h(f_1(x_1), g(x_2, \\ldots, x_n))\\\\
        \\\\\\\\
        x_i &\\in [0,1]
        \\end{align}

    Traditionally the problem is used with :math:`|x| = 30` dimensions in the solution space.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    def __init__(self, n=30, check_phenome=True):
        super().__init__(n, check_phenome)

    @property
    def bounds(self):
        return [(0, 1)] * self.n

    def evaluate(self, phenome):
        """
        :param phenome: contains x
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        if self.check_phenome:
            assert ((phenome >= 0) & (phenome <= 1)).all(),\
                f"All elements of phenome must be in [0.0, 1.0]: {phenome}"
        
        def g(x):
            """:param x: phenome[1..n]"""
            return 1 + 9 * (x.sum()) / (self.n - 1)
        
        fitness = np.zeros(2)
        fitness[0] = phenome[0]
        
        g_x = g(phenome[1:]) 
        h = 1 - (phenome[0] / g_x) ** 2
            
        fitness[1] = g_x * h

        return fitness


##############################
# Class ZDT3Problem
##############################
class ZDT3Problem(ZDTBenchmarkProblem):
    """
    The third problem from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite.  This function differs from :py:class:`leap_ec.problem.ZDT1Problem` and
    :py:class:`leap_ec.problem.ZDT1Problem` in that the pareto-optimal front has
    discontinuity.

    .. math::

        \\begin{align}
        f_1(x_1) &= x_1 \\\\
        g(x_2, \\ldots, x_n) &= 1 + 9\\cdot {\\sum_{i=2}^n x_i} / (n - 1) \\\\
        h(f_1, g) &= 1 - \\sqrt{f_1/g} - (f_1/g)\\sin(10\\pi f_1)\\\\
        f_2(x) &= g(x_2, \\ldots, x_n)h(f_1(x_1), g(x_2, \\ldots, x_n))\\\\
        \\\\\\\\
        x_i &\\in [0,1]
        \\end{align}

    Traditionally the problem is used with :math:`|x| = 10` dimensions in the solution space.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    def __init__(self, n=10, check_phenome=True):
        super().__init__(n, check_phenome)

    @property
    def bounds(self):
        return [(0, 1)] * self.n

    def evaluate(self, phenome):
        """
        :param phenome: contains x
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        if self.check_phenome:
            assert ((phenome >= 0) & (phenome <= 1)).all(),\
                f"All elements of phenome must be in [0.0, 1.0]: {phenome}"
        
        def g(x):
            """:param x: phenome[1..n]"""
            return 1 + 9 * (x.sum()) / (self.n - 1)
        
        fitness = np.zeros(2)
        fitness[0] = phenome[0]
        
        g_x = g(phenome[1:]) 
        h = 1 - np.sqrt(phenome[0] / g_x)\
            - (phenome[0] / g_x) * np.sin(10 * np.pi * phenome[0])
            
        fitness[1] = g_x * h

        return fitness


##############################
# Class ZDT4Problem
##############################
class ZDT4Problem(ZDTBenchmarkProblem):
    """
    The fourth problem from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite.  ZDT4 contains 21^9 local pareto-optimal front for the default parameters,
    allowing it to test for the EA's ability to handle multimodality.

    .. math::

        \\begin{align}
        f_1(x_1) &= x_1 \\\\
        g(x_2, \\ldots, x_n) &= 1 + 10(n-1) + \\sum_{i=2}^n (x_i^2 - 10\\cos(4\\pi x_i)) \\\\
        h(f_1, g) &= 1 - \\sqrt{f_1/g}\\\\
        f_2(x) &= g(x_2, \\ldots, x_n)h(f_1(x_1), g(x_2, \\ldots, x_n))\\\\
        \\\\\\\\
        x_1 &\\in [0,1] \\quad x_2, \\ldots, x_n \\in [-5,5]
        \\end{align}

    Traditionally the problem is used with :math:`|x| = 30` dimensions in the solution space.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    def __init__(self, n=30, check_phenome=True):
        super().__init__(n, check_phenome)

    @property
    def bounds(self):
        return [(0, 1)] + [(-5, 5)] * (self.n - 1)

    def evaluate(self, phenome):
        """
        :param phenome: contains x
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        if self.check_phenome:
            assert ((phenome[0] >= 0) and (phenome[0] <= 1)),\
                f"Element 0 of phenome must be in [0.0, 1.0]: {phenome}"
            assert ((phenome[1:] >= -5) & (phenome[1:] <= 5)).all(),\
                f"Elements 1 onward of phenome must be in [-5.0, 5.0]: {phenome}"
        
        def g(x):
            """:param x: phenome[1..n]"""
            return 1 + 10 * (self.n - 1) + np.sum(x ** 2 - 10 * np.cos(4 * np.pi * x))
        
        fitness = np.zeros(2)
        fitness[0] = phenome[0]
        
        g_x = g(phenome[1:])
        h = 1 - np.sqrt(phenome[0] / g_x)
            
        fitness[1] = g_x * h

        return fitness


##############################
# Class ZDT5Problem
##############################
class ZDT5Problem(ZDTBenchmarkProblem):
    """
    The fifth problem from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite.  In contrast to the other ZDT problems, ZDT5 takes a binary string as input.
    
    Unlike the other ZDT problems, `ZDT5Problem` additionally provides a `phenome_length`
    property, denoting the length of the flattened binary sequence `x`. This property is
    intended to ease the creation of binary sequence phenomes for input into the problem.

    .. math::

        \\begin{align}
        u(x_i) &= \\textrm{unitation}(x_i)\\\\
        v(u(x_i)) &= 
        \\left\\{
            \\begin{array}{lc}
                2+u(x_i) & if u(x_i) < 5 \\\\
                1 & if u(x_i) = 5 \\\\
            \\end{array}
        \\right\\}\\\\
        \\\\\\\\
        f_1(x_1) &= 1 + u(x_1) \\\\
        g(x_2, \\ldots, x_n) &= \\sum_{i=2}^n v(u(x_i)) \\\\
        h(f_1, g) &= 1 / f_1\\\\
        f_2(x) &= g(x_2, \\ldots, x_n)h(f_1(x_1), g(x_2, \\ldots, x_n))\\\\
        \\\\\\\\
        x_1 &\\in \\{0,1\\}^30 \\quad x_2, \\ldots, x_n \\in \\{0,1\\}^5
        \\end{align}

    Traditionally the problem is used with :math:`|x| = 11` dimensions in the solution space.
    This translates to a flattened binary sequence of :math:`|phenome_x| = 80`.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    def __init__(self, n=11, check_phenome=True):
        super().__init__(n, check_phenome)
        self._phenome_length = 30 + (self.n - 1) * 5

    @property
    def phenome_length(self):
        """
        :returns: the length of the flattened binary sequence x
        """
        return self._phenome_length
    
    @property
    def bounds(self):
        return [(0, 1)] * self._phenome_length
    
    def evaluate(self, phenome):
        """
        :param phenome: the flattened binary sequence x
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        
        if self.check_phenome:
            assert len(phenome) == self._phenome_length,\
                f"Phenome must be length {self._phenome_length}, actual length: {len(phenome)}"
            assert set(phenome).issubset({0, 1}),\
                f"Phenome must be a bit string: {phenome}"
        
        # Separate bit string into elements
        phenome = [
            phenome[:30],
            *(
                phenome[i:i+5] for i in range(30, len(phenome), 5)
            )
        ]
        
        def v(x_i):
            """:param x_i: phenome[i]"""
            u = np.sum(x_i)
            if u < 5:
                return 2 + u
            return 1
        
        def g(x):
            """:param x: phenome[1..n]"""
            return np.sum([v(x_i) for x_i in x])
        
        fitness = np.zeros(2)
        fitness[0] = 1 + np.sum(phenome[0])
        
        g_x = g(phenome[1:])
        h = 1 / fitness[0]
            
        fitness[1] = g_x * h

        return fitness


##############################
# Class ZDT6Problem
##############################
class ZDT6Problem(ZDTBenchmarkProblem):
    """
    The sixth problem from the classic Zitzler, Deb, and Thiele (ZDT) benchmark
    suite. This function exhibits a nonuniformly distributed pareto front, as well as
    a lower density of solutions nearer to the pareto front.

    .. math::

        \\begin{align}
        f_1(x_1) &= 1-\\textrm{exp}(-4x_1)\\sin^6(6\\pi x_1) \\\\
        g(x_2, \\ldots, x_n) &= 1 + 9\\cdot (({\\sum_{i=2}^n x_i}) / (n - 1))^{0.25} \\\\
        h(f_1, g) &= 1 - (f_1/g)^2\\\\
        f_2(x) &= g(x_2, \\ldots, x_n)h(f_1(x_1), g(x_2, \\ldots, x_n))\\\\
        \\\\\\\\
        x_i &\\in [0,1]
        \\end{align}

    Traditionally the problem is used with :math:`|x| = 10` dimensions in the solution space.

    - Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary
      algorithms: Empirical results." *Evolutionary computation* 8.2 (2000): 173-195.

    """
    def __init__(self, n=10, check_phenome=True):
        super().__init__(n, check_phenome)

    @property
    def bounds(self):
        return [(0, 1)] * self.n

    def evaluate(self, phenome):
        """
        :param phenome: contains x
        :returns: two fitnesses, one for :math:`f_1(x)` and :math:`f_2(x)`
        """
        if self.check_phenome:
            assert ((phenome >= 0) & (phenome <= 1)).all(),\
                f"All elements of phenome must be in [0.0, 1.0]: {phenome}"
        
        def g(x):
            """:param x: phenome[1..n]"""
            return 1 + 9 * ((x.sum()) / (self.n - 1)) ** 0.25
        
        fitness = np.zeros(2)
        fitness[0] = 1 - np.exp(-4 * phenome[0]) * np.sin(6 * np.pi * phenome[0]) ** 6
        
        g_x = g(phenome[1:])
        h = 1 - (fitness[0] / g_x) ** 2
            
        fitness[1] = g_x * h

        return fitness