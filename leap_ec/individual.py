#!/usr/bin/env python
"""
    Defines `Individual`
"""
from math import nan, isnan
from copy import deepcopy
from functools import total_ordering

from leap_ec.decoder import IdentityDecoder


##############################
# Class Individual
##############################
@total_ordering
class Individual:
    """
        Represents a single solution to a `Problem`.

        We represent an `Individual` by a `genome` and a `fitness`.
        `Individual` also maintains a reference to the `Problem` it will be
        evaluated on, and an `decoder`, which defines how genomes are
        converted into phenomes for fitness evaluation.
    """

    def __init__(self, genome, decoder=IdentityDecoder(), problem=None):
        """
        Initialize an `Individual` with a given genome.

        We also require `Individual`s to maintain a reference to the `Problem`:

        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.decoder import IdentityDecoder
        >>> ind = Individual([0, 0, 1, 0, 1], decoder=IdentityDecoder(), problem=MaxOnes())
        >>> ind.genome
        [0, 0, 1, 0, 1]

        Fitness defaults to `None`:

        >>> ind.fitness is None
        True

        :param genome: is the genome representing the solution.  This can be
            any arbitrary type that your mutation operators, probes, etc.,
            know how to read and manipulate---a list, class, etc.

        :param decoder: is a function or `callable` that converts a genome
            into a phenome.

        :param problem: is the `Problem` associated with this individual.
        """
        # Type checking to avoid difficult-to-debug errors
        if isinstance(decoder, type):
            raise ValueError(
                f"Got the type '{decoder}' as a decoder, but expected an instance.")
        if isinstance(problem, type):
            raise ValueError(
                f"Got the type '{problem}' as a problem, but expected an instance.")
        # Core data
        self.genome = genome
        self.problem = problem
        self.decoder = decoder
        self.fitness = None

    @classmethod
    def create_population(cls, n, initialize, decoder, problem):
        """
        A convenience method for initializing a population of the appropriate
        subtype.

        :param n: The size of the population to generate
        :param initialize: A function f(m) that initializes a genome
        :param decoder: The decoder to attach individuals to
        :param problem: The problem to attach individuals to
        :return: A list of n individuals of this class's (or subclass's) type
        """
        return [cls(genome=initialize(), decoder=decoder, problem=problem) for _
                in range(n)]

    @classmethod
    def evaluate_population(cls, population):
        """ Convenience function for bulk serial evaluation of a given
        population

        :param population: to be evaluated
        :return: evaluated population
        """
        for individual in population:
            individual.evaluate()

        return population

    def clone(self):
        """Create a 'clone' of this `Individual`, copying the genome, but not
        fitness.

        A deep copy of the genome will be created, so if your `Individual`
        has a custom genome type, it's important that it implements the
        `__deepcopy__()` method.

        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.decoder import IdentityDecoder
        >>> ind = Individual([0, 1, 1, 0], IdentityDecoder(), MaxOnes())
        >>> ind_copy = ind.clone()
        >>> ind_copy.genome == ind.genome
        True
        >>> ind_copy.problem == ind.problem
        True
        >>> ind_copy.decoder == ind.decoder
        True
        """
        new_genome = deepcopy(self.genome)
        cloned = type(self)(new_genome, self.decoder, self.problem)
        cloned.fitness = None
        return cloned

    def decode(self, *args, **kwargs):
        """
        :return: the decoded value for this individual
        """
        return self.decoder.decode(self.genome, args, kwargs)

    def evaluate_imp(self):
        """ This is the evaluate 'implementation' called by
            self.evaluate().   It's intended to be optionally over-ridden by
            sub-classes to give an opportunity to pass in ancillary data to
            the evaluate process either by tailoring the problem interface or
            that of the given decoder.
        """
        return self.problem.evaluate(self.decode())

    def evaluate(self):
        """ determine this individual's fitness

        This is done by outsourcing the fitness evaluation to the associated
        Problem object since it "knows" what is good or bad for a given
        phenome.


        :see also: ScalarProblem.worse_than

        :return: the calculated fitness
        """
        self.fitness = self.evaluate_imp()
        return self.fitness

    def __iter__(self):
        """
        :raises: exception if self.genome is None
        :return: the encapsulated genome's iterator
        """
        return self.genome.__iter__()

    def __eq__(self, other):
        """
        Note that the associated problem knows best how to compare associated
        individuals

        :param other: to which to compare
        :return: if this Individual is less fit than another
        """
        if other is None:  # Never equal to None
            return False
        return self.problem.equivalent(self.fitness, other.fitness)

    def __lt__(self, other):
        """
        Because `Individual`s know about their `Problem`, they know how to
        compare themselves to one another.  One individual is better than
        another if and only if it is greater than the other:

        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.decoder import IdentityDecoder
        >>> f = MaxOnes(maximize=True)
        >>> ind_A = Individual([0, 0, 1, 0, 1], IdentityDecoder(), problem=f)
        >>> ind_A.fitness = 2
        >>> ind_B = Individual([1, 1, 1, 1, 1], IdentityDecoder(), problem=f)
        >>> ind_B.fitness = 5
        >>> ind_A > ind_B
        False


        Use care when writing selection operators! When comparing
        `Individuals`, `>` always means "better than." The `>` function may
        indicate maximization, minimization, Pareto dominance, etc.: it all
        depends on the underlying `Problem`.

        >>> f = MaxOnes(maximize=False)
        >>> ind_A = Individual([0, 0, 1, 0, 1], IdentityDecoder(), problem=f)
        >>> ind_A.fitness = 2
        >>> ind_B = Individual([1, 1, 1, 1, 1], IdentityDecoder(), problem=f)
        >>> ind_B.fitness = 5
        >>> ind_A > ind_B
        True

        Note that the associated problem knows best how to compare associated
        individuals

        :param other: to which to compare
        :return: if this Individual has the same fitness as another even if
                 they have different genomes
        """
        if other is None:  # Always better than None
            return False
        return self.problem.worse_than(self.fitness, other.fitness)

    def __str__(self):
        return f'{self.genome!s} {self.fitness!s}'

    def __repr__(self):
        return f"{type(self).__name__}({self.genome.__repr__()}, " \
               f"{self.decoder.__repr__()}, {self.problem.__repr__()})"


##############################
# Class RobustIndividual
##############################
class RobustIndividual(Individual):
    """
        This adds exception handling for evaluations

        After evaluation `self.is_viable` is set to True if all went well.
        However, if an exception is thrown during evaluation, the following
        happens:

        * self.is_viable is set to False
        * self.fitness is set to math.nan
        * self.exception is assigned the exception
    """
    def __init__(self, genome, decoder=None, problem=None):
        super().__init__(genome, decoder=decoder, problem=problem)

    def evaluate(self):
        """ determine this individual's fitness

        Note that if an exception is thrown during evaluation, the fitness is
        set to NaN and `self.is_viable` to False; also, the returned exception is
        assigned to `self.exception` for possible later inspection.  If the
        individual was successfully evaluated, `self.is_viable` is set to true.
        NaN fitness values will figure into comparing individuals in that NaN
        will always be considered worse than non-NaN fitness values.

        :return: the calculated fitness
        """
        try:
            self.fitness = self.evaluate_imp()
            self.is_viable = True  # we were able to evaluate
        except Exception as e:
            self.fitness = nan
            self.exception = e
            self.is_viable = False  # we could not complete an eval

        # Even though we've already *set* the fitness, it may be useful to also
        # *return* it to give more options to the programmer for using the
        # newly evaluated fitness.
        return self.fitness
