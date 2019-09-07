"""
    Classes related to individuals that represent posed solutions.


    TODO Need to decide if the logic in __eq__ and __lt__ is overly complex.
    I like that this reduces the dependency on Individuals on a Problem
    (because sometimes you have a super simple situation that doesn't require
    explicitly couching your problem in a Problem subclass.)
"""
from copy import deepcopy
from functools import total_ordering


@total_ordering
class Individual:
    """
        Represents a single solution to a `Problem`.

        We represent an `Individual` by a `genome`, a `fitness`, and an optional dict of `attributes`.

        `Individual` also maintains a reference to the `Problem` it will be evaluated on, and an `decoder`, which
        defines how genomes are converted into phenomes for fitness evaluation.
    """

    def __init__(self, genome, decoder=None, problem=None):
        """
        Initialize an `Individual` with a given genome.

        We also require `Individual`s to maintain a reference to the `Problem`:

        >>> from leap import decode, binary
        >>> ind = Individual([0, 0, 1, 0, 1], decoder=decode.IdentityDecoder(), problem=binary.MaxOnes())
        >>> ind.genome
        [0, 0, 1, 0, 1]

        Fitness defaults to `None`:

        >>> ind.fitness is None
        True

        :param genome: is the genome representing the solution.  This can be any arbitrary type that your mutation
            operators, probes, etc., know how to read and manipulate---a list, class, etc.
        :param decoder: is a function or `callable` that converts a genome into a phenome.
        :param problem: is the `Problem` associated with this individual.
        """
        # Core data
        self.genome = genome
        self.problem = problem
        self.decoder = decoder
        self.fitness = None

        # A dict to hold application-specific attributes
        self.attributes = dict()

    def clone(self):
        """Create a 'clone' of this `Individual`, copying the genome, but not fitness.

        A deep copy of the genome will be created, so if your `Individual` has a custom genome type, it's important
        that it implements the `__deepcopy__()` method.

        >>> from leap import decode, binary
        >>> ind = Individual([0, 1, 1, 0], decode.IdentityDecoder(), binary.MaxOnes())
        >>>
        """
        new_genome = deepcopy(self.genome)
        cloned = type(self)(new_genome, self.decoder, self.problem)
        cloned.fitness = None
        return cloned

    def decode(self):
        """
        :return: the decoded value for this individual
        """
        return self.decoder.decode(genome=self.genome)

    def evaluate(self):
        self.fitness = self.problem.evaluate(self.decode())

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
        return self.problem.equivalent(self.fitness, other.fitness)

    def __lt__(self, other):
        """
        Because `Individual`s know about their `Problem`, the know how to compare themselves
        to one another.  One individual is better than another if and only if it is greater than the other:

        >>> from leap import decode, binary
        >>> f = binary.MaxOnes(maximize=True)
        >>> ind_A = Individual([0, 0, 1, 0, 1], decode.IdentityDecoder, problem=f)
        >>> ind_A.fitness = 2
        >>> ind_B = Individual([1, 1, 1, 1, 1], decode.IdentityDecoder, problem=f)
        >>> ind_B.fitness = 5
        >>> ind_A > ind_B
        False


        Use care when writing selection operators! When comparing `Individuals`, `>` always means "better than."
        The `>` function may indicate maximization, minimization, Pareto dominance, etc.: it all depends on the
        underlying `Problem`.

        >>> f = binary.MaxOnes(maximize=False)
        >>> ind_A = Individual([0, 0, 1, 0, 1], decode.IdentityDecoder, problem=f)
        >>> ind_A.fitness = 2
        >>> ind_B = Individual([1, 1, 1, 1, 1], decode.IdentityDecoder, problem=f)
        >>> ind_B.fitness = 5
        >>> ind_A > ind_B
        True

        Note that the associated problem knows best how to compare associated
        individuals

        :param other: to which to compare
        :return: if this Individual has the same fitness as another even if
                 they have different genomes
        """
        return self.problem.worse_than(self.fitness, other.fitness)

    def __repr__(self):
        # TODO Is this the right behavior for __repr__() vs. __str__()?
        return self.genome.__repr__()
