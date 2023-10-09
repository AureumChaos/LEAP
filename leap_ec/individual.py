#!/usr/bin/env python
"""
    Defines `Individual`
"""
from math import nan
from copy import copy, deepcopy
from functools import total_ordering
import uuid

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
        Initialize an `Individual` with a given genome.  A UUID is generated
        and assigned to `self.uuid`.  The `parents` set is initialized to be
        empty.

        We also require `Individual`s to maintain a reference to the `Problem`:

        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.decoder import IdentityDecoder
        >>> import numpy as np
        >>> genome = np.array([0, 0, 1, 0, 1])
        >>> ind = Individual(genome, decoder=IdentityDecoder(),
        ...                  problem=MaxOnes())
        >>> ind.genome
        array([0, 0, 1, 0, 1])

        Fitness defaults to `None`:

        >>> ind.fitness is None
        True

        :param genome: is the genome representing the solution.  This can be
            any arbitrary type that your mutation operators, probes, etc.,
            know how to read and manipulate---a list, class, numpy array, etc.

        :param decoder: is a function or `callable` that converts a genome
            into a phenome.

        :param problem: is the `Problem` associated with this individual.
        """
        # Type checking to avoid difficult-to-debug errors
        if isinstance(decoder, type):
            raise ValueError((
                f"Got the type '{decoder}' as a decoder, but expected an"
                " instance."))
        if isinstance(problem, type):
            raise ValueError((
                f"Got the type '{problem}' as a problem, but expected an"
                " instance."))
        # Core data
        self.genome = genome
        self.problem = problem
        self.decoder = decoder
        self.fitness = None
        self._phenome = None
        
        self.uuid = uuid.uuid4() # every individual gets a unique ID
        self.parents = set() # set of uuids of parents


    @property
    def phenome(self):
        """If the phenome has not yet been decoded, do so."""
        if self._phenome is None:
            self.decode()
        return self._phenome

    @phenome.setter
    def phenome(self, value):
        """Manually set the phenome, bypassing the decoder."""
        self._phenome = value

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
        return [cls(genome=initialize(), decoder=decoder, problem=problem)
                for _ in range(n)]

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

        The fitness of the clone is set to `None`.  A new UUID is generated and
        assigned to `sefl.uuid`.  The `parents` set is updated to include the
        UUID of the parent.  A shallow copy of the parent is made, too, so that
        ancillary state is also copied.

        A deep copy of the genome will be created, so if your `Individual`
        has a custom genome type, it's important that it implements the
        `__deepcopy__()` method.

        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.decoder import IdentityDecoder
        >>> import numpy as np
        >>> genome = np.array([0, 1, 1, 0])
        >>> ind = Individual(genome, IdentityDecoder(), MaxOnes())
        >>> ind_copy = ind.clone()
        >>> ind_copy.genome == ind.genome
        array([ True,  True,  True,  True])
        >>> ind_copy.problem == ind.problem
        True
        >>> ind_copy.decoder == ind.decoder
        True
        """
        cloned = copy(self)
        cloned.genome = deepcopy(self.genome)
        cloned.fitness = None

        cloned.uuid = uuid.uuid4()
        cloned.parents = {self.uuid}

        return cloned

    def decode(self, *args, **kwargs):
        """
        Determine the indivdual's phenome.

        This is done by passing the genome `self.decoder`.

        The result is both returned and saved to `self.phenome`.

        :return: the decoded value for this individual
        """
        self._phenome = self.decoder.decode(self.genome, args, kwargs)
        return self._phenome

    def evaluate_imp(self):
        """ This is the evaluate 'implementation' called by
            self.evaluate().   It's intended to be optionally over-ridden by
            sub-classes to give an opportunity to pass in ancillary data to
            the evaluate process either by tailoring the problem interface or
            that of the given decoder.
        """
        self.decode()
        return self.problem.evaluate(self.phenome)

    def evaluate(self):
        """ determine this individual's fitness

        This is done by outsourcing the fitness evaluation to the associated
        `Problem` object since it "knows" what is good or bad for a given
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
        assert(hasattr(other, 'fitness')), f"Object {other} has no 'fitness' attribute."
        return self.problem.equivalent(self.fitness, other.fitness)

    def __lt__(self, other):
        """
        Because `Individual`s know about their `Problem`, they know how to
        compare themselves to one another.  One individual is better than
        another if and only if it is greater than the other:

        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.decoder import IdentityDecoder
        >>> import numpy as np
        >>> f = MaxOnes(maximize=True)
        >>> genome_A = np.array([0, 0, 1, 0, 1])
        >>> ind_A = Individual(genome_A, IdentityDecoder(), problem=f)
        >>> ind_A.fitness = 2
        >>> genome_B = np.array([1, 1, 1, 1, 1])
        >>> ind_B = Individual(genome_B, IdentityDecoder(), problem=f)
        >>> ind_B.fitness = 5
        >>> ind_A > ind_B
        False


        Use care when writing selection operators! When comparing
        `Individuals`, `>` always means "better than." The `>` function may
        indicate maximization, minimization, Pareto dominance, etc.: it all
        depends on the underlying `Problem`.

        >>> f = MaxOnes(maximize=False)
        >>> ind_A = Individual(genome_A, IdentityDecoder(), problem=f)
        >>> ind_A.fitness = 2
        >>> ind_B = Individual(genome_B, IdentityDecoder(), problem=f)
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
        return f'{type(self).__name__}<{self.uuid}> with fitness {self.fitness!s}'

    def __repr__(self):
        return f"{type(self).__name__}<{self.uuid}>({self.genome.__repr__()}, " \
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
    def __init__(self, genome, decoder=IdentityDecoder(), problem=None):
        super().__init__(genome, decoder=decoder, problem=problem)

    def evaluate(self):
        """ determine this individual's fitness

        Note that if an exception is thrown during evaluation, the fitness is
        set to NaN and `self.is_viable` to False; also, the returned exception
        is assigned to `self.exception` for possible later inspection.  If the
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


##############################
# WholeEvalautedIndividual
##############################
class WholeEvaluatedIndividual(Individual):
    """An Individual that, when evaluated, passes its whole self
    to the evaluation function, rather than just its phenome.
    
    In most applications, fitness evaluation requires only phenome
    information, so that is all that we pass from the Individual to the
    Problem.  This is important, because during distributed evaluation,
    we want to pass as little information as possible across nodes.

    WholeEvaluatedIndividual is used for special cases where fitness
    evaluation needs access to more information about an individual than
    its phenome.  This is strange in most cases and should be avoided,
    but can make certain algorithms more elegant (ex. it's helpful when
    interpreting cooperative coevolution as an island model).
    
    This can dramatically slow down distributed evaluation (i.e. with dask)
    in some applications because the entire individual will be sent over a
    TCP/IP connection instead of just the `phenome`, so use with caution.
    """
    def evaluate_imp(self):
        """ This is the evaluate 'implementation' called by
            self.evaluate().   It's intended to be optionally over-ridden by
            sub-classes to give an opportunity to pass in ancillary data to
            the evaluate process either by tailoring the problem interface or
            that of the given decoder.
        """
        self.decode()
        return self.problem.evaluate(self.phenome, individual=self)