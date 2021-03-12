"""
Defines the abstract-base classes Problem, ScalarProblem,
and FunctionProblem.

"""
from abc import ABC, abstractmethod
from math import nan, floor
import random

import numpy as np

from leap_ec.context import context


##############################
# Class Problem
##############################
class Problem(ABC):
    """
        Abstract Base Class used to define problem definitions.

        A `Problem` is in charge of two major parts of an EA's behavior:

         1. Fitness evaluation (the `evaluate()` method)

         2. Fitness comparision (the `worse_than()` and `equivalent()` methods)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate(self, phenome, *args, **kwargs):
        """
        Decode and evaluate the given individual based on its genome.

        Practitioners *must* over-ride this member function.

        Note that by default the individual comparison operators assume a
        maximization problem; if this is a minimization problem, then just
        negate the value when returning the fitness.

        :param phenome:
        :return: fitness
        """
        raise NotImplementedError

    @abstractmethod
    def worse_than(self, first_fitness, second_fitness):
        raise NotImplementedError

    @abstractmethod
    def equivalent(self, first_fitness, second_fitness):
        raise NotImplementedError


##############################
# Class ScalarProblem
##############################
class ScalarProblem(Problem):
    """A problem that compares individuals based on their scalar fitness values.
    
    Inherit from this class and implement the `evaluate()` method to implement
    an objective function that returns a single real-valued fitness value.
    """
    def __init__(self, maximize):
        super().__init__()
        self.maximize = maximize

    def worse_than(self, first_fitness, second_fitness):
        """
            Used in Individual.__lt__().

            By default returns first_fitness < second_fitness if a maximization
            problem, else first_fitness > second_fitness if a minimization
            problem.  Please over-ride if this does not hold for your problem.

            If both fitnesses are nan, a random Boolean is returned.

            :return: true if the first individual is less fit than the second
        """
        # NaN is assigned if the individual is non-viable, which can happen if
        # an exception is thrown during evaluation. We consider NaN fitnesses to
        # always be the worse possible with regards to ordering.

        # XXX This seems like logic that was specific to a particular application.
        # XXX It seems surprising here.  Move elsewhere, and perhaps replace with
        # XXX something like assert(first_fitness is not nan)? -Siggy

        if first_fitness is nan:
            if second_fitness is nan:
                # both are nan, so to reduce bias bitflip a coin to arbitrarily
                # select one that is worst.
                return random.choice([True, False])
            # Doesn't matter how awful second_fitness is, nan will already be
            # considered worse.
            return True
        elif second_fitness is nan:
            # No matter how awful the first_fitness is, if it's not a NaN the
            # NaN will always be worse
            return False

        # TODO If we accidentally pass an Individual in as first_ or second_fitness,
        # TODO then this can result in an infinite loop.  Add some error
        # handling for this.
        if self.maximize:
            return first_fitness < second_fitness
        else:
            return first_fitness > second_fitness

    def equivalent(self, first_fitness, second_fitness):
        """
            Used in Individual.__eq__().

            By default returns first.fitness== second.fitness.  Please
            over-ride if this does not hold for your problem.

            :return: true if the first individual is equal to the second
        """
        if first_fitness is nan and second_fitness is nan:
            return True

        # TODO Should we consider (abs(first_fitness-second_fitness) < epsilon)
        return first_fitness == second_fitness


##############################
# Class MultiObjectiveProblem
##############################
class MultiObjectiveProblem(Problem):
    """A problem that compares individuals based on Pareto dominance.
    
    Inherit from this class and implement the `evaluate()` method to implement
    an objective function that returns a list of real-value fitness values.

    In Pareto-dominance, an individual A is only considered "better than" an individual
    B if A is unamibiguously better than B: i.e. it is at least as good as B on
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
        assert(maximize is not None)
        assert(len(maximize) > 0)
        # Represent maximize as a vector of 1's and -1's
        self.maximize = 1 * np.array(maximize) - 1 * np.invert(maximize)

    def worse_than(self, first_fitness, second_fitness):
        """Return true if first_fitness is Pareto-dominated by second_fitness.

        In the case of maximization over all objectives, a solution :math:`b` 
        dominates :math:`a`, written :math:`b \succ a`, if and only if

        .. math::

              \\begin{array}{ll}
                f_i(b) \\ge f_i(a) & \\forall i, \\text{ and} \\\\
                f_i(b) > f_j(a) & \\text{ for some } j.
              \\end{array}
          
        Here we may maximize over some objectives, and minimize over others, 
        depending on the values in the `self.maximize` list.

        """
        assert(first_fitness is not None)
        assert(second_fitness is not None)
        assert(len(first_fitness) == len(self.maximize))
        assert(len(second_fitness) == len(self.maximize))

        # Negate the minimization problems, so we can treat all objectives as maximization
        first_max = first_fitness * self.maximize
        second_max = second_fitness * self.maximize

        # Now check the two conditions for dominance using numpy comparisons
        return all (second_max >= first_max) \
                and any (second_max > first_max)

    def equivalent(self, first_fitness, second_fitness):
        """Return true if first_fitness and second_fitness are mutually
        Pareto non-dominating.

        .. math::
            a \\not \\succ b \\text{ and } b \\not \\succ a

        """
        return not self.worse_than(first_fitness, second_fitness) \
            and not self.worse_than(second_fitness, first_fitness)


##############################
# Class FunctionProblem
##############################
class FunctionProblem(ScalarProblem):
    """A convenience wrapper that takes a vanilla function that returns scalar
    fitness values and makes it usable as an objective function."""
    def __init__(self, fitness_function, maximize):
        super().__init__(maximize)
        self.fitness_function = fitness_function

    def evaluate(self, phenome, *args, **kwargs):
        return self.fitness_function(phenome, *args, **kwargs)


##############################
# Class ConstantProblem
##############################
class ConstantProblem(ScalarProblem):
    """A flat landscape, where all phenotypes have the same fitness.

    This is sometimes useful for sanity checks or as a control in certain
    kinds of research.

    .. math::

       f(\\vec{x}) = c

    :param float c: the fitness value to return for any input.

    .. plot::
       :include-source:

       from leap_ec.problem import ConstantProblem
       from leap_ec.real_rep.problems import plot_2d_problem
       bounds = ConstantProblem.bounds
       plot_2d_problem(ConstantProblem(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """Default bounds."""
    bounds = (-1.0, 1.0)

    def __init__(self, maximize=False, c=1.0):
        super().__init__(maximize)
        self.c = c

    def evaluate(self, phenome, *args, **kwargs):
        """
        Return a contant value for any input phenome:

        >>> phenome = [0.5, 0.8, 1.5]
        >>> ConstantProblem().evaluate(phenome)
        1.0

        >>> ConstantProblem(c=500.0).evaluate('foo bar')
        500.0

        :param phenome: real-valued vector to be evaluated
        :return: 1.0, or the constant defined in the constructor
        """
        return self.c

    def __str__(self):
        return ConstantProblem.__name__


########################
# Class AlternatingProblem
########################
class AlternatingProblem(Problem):
    def __init__(self, problems, modulo, context=context):
        assert(len(problems) > 0)
        assert(modulo > 0)
        assert(context is not None)
        self.problems = problems
        self.modulo = modulo
        self.context = context
        self.current_problem_idx = 0

    def _get_current_problem(self):
        assert('leap' in self.context)
        assert('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        i = floor(step / self.modulo) % len(self.problems)

        return self.problems[i]

    def evaluate(self, phenome):
        return self._get_current_problem().evaluate(phenome)

    def worse_than(self, first_fitness, second_fitness):
        return self._get_current_problem().worse_than(first_fitness, second_fitness)

    def equivalent(self, first_fitness, second_fitness):
        return self._get_current_problem().equivalent(first_fitness, second_fitness)
    

##############################
# Class MultiObjectiveToolkitProblem
##############################
class MultiObjectiveToolkitProblem(MultiObjectiveProblem):
    """A problem that implements Kalyanmoy Deb's popular tunable two-objective problem 'toolkit.'
    
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
    """
    def __init__(self, f1, f1_input_length: int, g, h, maximize: list):
        assert(f1 is not None)
        assert(callable(f1))
        assert(f1_input_length > 0)
        assert(g is not None)
        assert(callable(g))
        assert(h is not None)
        assert(callable(h))
        super().__init__(maximize)
        self.f1 = f1
        self.f1_input_length = f1_input_length
        self.g = g
        self.h = h

    def evaluate(self, phenome, *args, **kwargs):
        y = phenome[:self.f1_input_length]
        z = phenome[self.f1_input_length:]

        o1 = self.f1(y)
        g_out = self.g(z)
        o2 = g_out * h(o1, g_out)
        return (o1, o2)
