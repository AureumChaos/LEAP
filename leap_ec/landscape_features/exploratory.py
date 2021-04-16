"""This module implements features that are common in exploratory landscape analysis (ELA).

The gist of exploratory landscape analysis is that it provides a set of 

    1. statistical features for measuring properties of continuous fitness landscapes, with
    2. an emphasis on using a small number of fitness samples to compute the features.

The big idea is that these are features you can use to measure and understand a problem
before solving it.

There are dozens of traditional ELA features.  Most are described in the following two 
seminal papers:

 * Mersmann, Olaf, et al. "`Exploratory landscape analysis <https://dl.acm.org/doi/abs/10.1145/2001576.2001690>`_."
   *Proceedings of the 13th annual conference on Genetic and evolutionary computation.* 2011.

 * Kerschke, Pascal, et al. "`Cell mapping techniques for exploratory landscape analysis
   <https://link.springer.com/chapter/10.1007/978-3-319-07494-8_9>`_." *EVOLVE-A Bridge between Probability, Set
   Oriented Numerics, and Evolutionary Computation V.* Springer, Cham, 2014. 115-131.

"""
import numpy as np
import pandas as pd

from leap_ec.individual import Individual


class ELAConvexity():
    """
    This class provides features that empirically estimate the degree to which a landscape is convex or linear.

    :param problem: the fitness landscape to analyze (must accept real-vector phenomes).
    :param representation: a :class:`Representation` that can be used to sample and decode new individuals (must decode
        individuals into a real-vector phenome).
    :param design_individuals: an initial sample individuals that is used as the basis for analysis (their fitnesses
        must already have been evaluated).
    :param int num_convexity_tests: the number of pairwises tests (and additional fitness samples) to use in
        estimating convexity features.

    The algorithm used here is best explained by the following graphic:

    .. plot::

        from matplotlib import pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches
            
        plt.figure(figsize=(12, 10))

        # Line
        plt.plot([1.0, 2.0], [1.0, 2.0], marker='o', markersize=10, color='black', linestyle='dashed')
        plt.annotate("x", (1.05, 0.9), fontsize='xx-large')
        plt.annotate("y", (2.05, 1.9), fontsize='xx-large')

        # Convex combination
        plt.plot([1.5], [1.5], marker='o', markersize=10, color='blue')
        plt.annotate("comb(x, y)", (1.2, 1.55), fontsize='xx-large')

        # Third point
        plt.plot([1.5], [1.1], marker='o', markersize=10, color='blue')
        plt.annotate('p', (1.55, 1.0), fontsize='xx-large')

        # Suggested convex surface
        vertices = [
            (1.0, 1.0), # Start point
            (1.2, 1.0), # Beizer control point
            (1.5, 1.1), # Beizer midpoint
            (1.8, 1.2), # Beizer control point
            (2.0, 2.0)  # End point
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.LINETO
        ]

        path = Path(vertices, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        plt.gca().add_patch(patch)

        plt.annotate("Suggested\\nconvex surface", (1.85, 1.3), fontsize='xx-large')


        plt.xlim(0.5, 2.5)
        plt.ylim(0.5, 2.5)
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.xlabel('Genotype', fontsize=20)
        plt.ylabel('Fitness', fontsize=20)
        plt.title("Convexity Sample Illustration", fontsize=20);

    We take a number of random pairs :math:`x, y` of individuals from the initial design (sample), and compute a
    third point :math:`p` from them via a `convex combination <https://en.wikipedia.org/wiki/Convex_combination>`_
    of the original two points (i.e. a random point lying along the line between the original points).  Then we
    compare the fitness :math:`f(p)` of the new point to the fitness that the point *would* have if the landscape were
    perfectly linear between the original points.  That is, we compute the *difference* between the complex
    combination of the parents' *fitness values* :math:`f(x)` and :math:`f(y)` and the fitness value of their
    *genomes'* complex combination, :math:`f(p)`.

    .. math::

        \\delta = f(p) - \\text{comb}(f(x), f(y))


    When :math:`\\delta \\approx 0`, it suggests local linearity of the fitness landscape, whereas when :math:`\\delta < 0`,
    convexity is suggested.

    To compute these features, we'll need a problem and a representation:

    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.individual import Individual
    >>> from leap_ec.representation import Representation
    >>> from leap_ec.real_rep import initializers, problems

    >>> DIMENSIONS = 10
    >>> N_SAMPLES = 50*DIMENSIONS
    >>> problem = problems.SpheroidProblem()

    >>> representation = Representation(
    ...     initialize=initializers.create_real_vector(bounds=[(-5.12, 5.12)]*DIMENSIONS)
    ... )

    We'll also need an initial sample of individuals must be provided, with its 
    fitnesses already evaluated:

    >>> initial_sample = representation.create_population(N_SAMPLES, problem)
    >>> initial_sample = Individual.evaluate_population(initial_sample);

    The feature computation uses this as its initial "experiment design," and then takes additional 
    fitness samples as needed when we call the constructor:

    >>> convex = ELAConvexity(problem, representation, design_individuals=initial_sample)

    The resulting object can be used to compute the various feature calculations:

    >>> x = convex.convex_p()

    >>> x = convex.linear_p()

    >>> x = convex.linear_deviation()

    """
    def __init__(self, problem, representation, design_individuals: list, num_convexity_tests: int=1000):
        assert(problem is not None)
        assert(representation is not None)
        assert(design_individuals is not None)
        assert(len(design_individuals) > 2)
        assert(num_convexity_tests > 0)
        self.problem = problem
        self.representation = representation
        self.design_individuals = design_individuals
        self.num_convexity_tests = num_convexity_tests

        # Take additional fitness samples for each convexity test
        self.pairs_list, self.combinations_list, self.deltas_list = self._compute_deltas()
    
    @property
    def pairs(self):
        """Contains all of the pairs of original individuals that were used in
        the convexity tests.
        """
        return self.pairs_list[:]  # Defensive copy

    @property
    def combinations(self):
        """Contains the list of `(f, p)` pairs, where `f` is the convex combination
        of the fitness pair that was used in the ith test, and `p` is the individual
        formed from the convex combination of their genomes.
        """
        return self.combinations_list[:]  # Defensive copy

    @property
    def deltas(self):
        """Contains the list of :math:`\\delta = f(p) - \\text{comb}(f(x), f(y))` values
        that were computed for the convexity tests.
        """
        return self.deltas_list[:]  # Defensive copy

    def _compute_deltas(self):
        """Sample additional points by create a convex combination of random
        pairs of individuals in the original design.

        Returns a list of :math:`\\delta` values, calculated as :math:`\\delta = f(p) - \\text{comb}(f(x), f(y))`,
        where :math:`p` is the individual
        resulting from the complex combination of the pair, a :math:`\\text{comb}(f(x), f(y))` is the
        complex combination of the *objective values* for each pair.
        """
        pairs = []
        combinations = []
        deltas = []
        for _ in range(self.num_convexity_tests):
            # Choose two individuals from the initial experiment design without replacement
            # (so the same individual is not chosen twice)
            x, y = np.random.choice(self.design_individuals, size=2, replace=False)
            pairs.append((x, y))

            # Find the convex combination of the original fitnesses
            a = np.random.uniform(0, 1)
            b = 1.0 - a
            f = a*x.fitness + b*y.fitness

            # Sample another point by taking the same convex combination of x and y
            x_genome = x.genome
            y_genome = y.genome
            p_genome = a*x_genome + b*y_genome

            # Evalute the new point's fitness
            p = Individual(p_genome, problem=self.problem, decoder=self.representation.decoder)  # Need a decoder; can't assume IdentityDecoder
            p.evaluate()
            combinations.append((f, p))

            delta = p.fitness - f
            deltas.append(delta)

        return pairs, combinations, deltas

    def convex_p(self, threshold: float=-0.0000000001):
        """Estimate the probability that the landscape is convex by calculating the frequency
        with which 

        .. math::

            \\delta < \\tau

        where :math:`\\tau` is a small negative threshold (typically :math:`-10^{-10}`).

        :param float threshold: the value of :math:`\\tau`.
        """
        assert(threshold < 0)
        lessthan_count = 0
        for d in self.deltas:
            if d < threshold:
                lessthan_count += 1
        
        ratio = lessthan_count/len(self.deltas)
        assert(ratio >= 0.0)
        assert(ratio <= 1.0)
        return ratio

    def linear_p(self, threshold: float=-0.0000000001):
        """Estimate the probability that the landscape is linear by calculating the frequency
        with which

        .. math::

            | \\delta | < \\tau

        where :math:`\\tau` is a small negative threshold (typically :math:`-10^{-10}`).

        :param float threshold: the value of :math:`\\tau`.
        """
        assert(threshold < 0)
        lessthan_count = 0
        for d in self.deltas:
            if np.abs(d) < threshold:
                lessthan_count += 1
        
        ratio = lessthan_count/len(self.deltas)
        assert(ratio >= 0.0)
        assert(ratio <= 1.0)
        return ratio

    def linear_deviation(self):
        """Estimate the deviation of the landscape from linearity by averaging the :math:`\\delta`
        values."""
        return np.mean(self.deltas)

    def linear_deviation_abs(self):
        """Estimate the deviation of the landscape of linearity by averagin the absolute value
        :math:`|\\delta|` of the computed deltas.
        
        Sometimes this is simply the negative of linear_deviation() (ex. when the function is
        completely convex), but other times the two values differ considerably."""
        return np.mean(np.abs(self.deltas))

    def results_table(self, function_name=None):
        """Return a Pandas dataframe as a convenience, with one row for each computed feature."""
        if function_name is None:
            function_name = str(self.problem)

        return pd.DataFrame([{ 'Function': function_name,
                               'P(convex)': self.convex_p(),
                               'P(linear)': self.linear_p(),
                               'Linear_deviation': self.linear_deviation(),
                               'Linear_deviation_abs': self.linear_deviation_abs()
                            }])
