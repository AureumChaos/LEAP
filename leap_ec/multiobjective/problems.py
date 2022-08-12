#!/usr/bin/env python3
"""
    LEAP Problem classes for multiobjective optimization.
"""


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

    This framework is used to define several classic multi-objective benchmark problems,
    such as :py:class:`leap_ec.real_rep.problems.ZDT1Problem`, etc.

    - Deb, Kalyanmoy. "Multi-objective genetic algorithms: Problem difficulties and
      construction of test problems." *Evolutionary computation* 7.3 (1999): 205-230.
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

    def evaluate(self, Individual, *args, **kwargs):
        phenome = Individual.phenome

        y = phenome[:self.f1_input_length]
        z = phenome[self.f1_input_length:]

        o1 = self.f1(y)
        g_out = self.g(z)
        o2 = g_out * h(o1, g_out)
        return (o1, o2)
