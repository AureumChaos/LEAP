"""
    Classes related to individuals that represent posed solutions.


    TODO Need to decide if the logic in __eq__ and __lt__ is overly complex.
    I like that this reduces the dependency on Individuals on a Problem
    (because sometimes you have a super simple situation that doesn't require
    explicitly couching your problem in a Problem subclass.)
"""
import abc
from copy import deepcopy
from functools import total_ordering
import random

from toolz import curry
from toolz.itertoolz import pluck

# This defines a global context that is a dictionary of dictionaries.  The
# intent is for certain operators and functions to add to and modify this
# context.  Third party operators and functions will just add a new top-level
# dedicated key.
context = {'leap': {}}


@curry
def create_binary_sequence(length=5):
    """ for creating a binary sequences for binary genomes

    E.g., can be used for Individual.create_population

    :param length: how many genes?
    :return: binary vector of given length
    """
    return [random.choice([0, 1]) for _ in range(length)]


##############################
# Class Individual
##############################
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

        >>> from leap import binary_problems
        >>> ind = Individual([0, 0, 1, 0, 1], decoder=IdentityDecoder(), problem=binary_problems.MaxOnes())
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

    @classmethod
    def create_population(cls, n, initialize, decoder, problem):
        """
        A convenience method for initializing a population of the appropriate subtype.

        :param n: The size of the population to generate
        :param initialize: A function f(m) that initializes a genome
        :param decoder: The decoder to attach individuals to
        :param problem: The problem to attach individuals to
        :return: A list of n individuals of this class's (or subclass's) type
        """
        # genomes = initialize(n)
        # assert(len(genomes) == n)
        return [cls(genome=initialize(), decoder=decoder, problem=problem) for _ in range(n)]

    @classmethod
    def evaluate_population(cls, population):
        """ Convenience function for bulk serial evaluation of a given population

        :param population: to be evaluated
        :return: evaluated population
        """
        for individual in population:
            individual.evaluate()

        return population

    def clone(self):
        """Create a 'clone' of this `Individual`, copying the genome, but not fitness.

        A deep copy of the genome will be created, so if your `Individual` has a custom genome type, it's important
        that it implements the `__deepcopy__()` method.

        >>> from leap import binary_problems
        >>> ind = Individual([0, 1, 1, 0], IdentityDecoder(), binary_problems.MaxOnes())
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
        return self.decoder.decode(self.genome)

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
        if other is None:  # Never equal to None
            return False
        return self.problem.equivalent(self.fitness, other.fitness)

    def __lt__(self, other):
        """
        Because `Individual`s know about their `Problem`, the know how to compare themselves
        to one another.  One individual is better than another if and only if it is greater than the other:

        >>> from leap import binary_problems
        >>> f = binary_problems.MaxOnes(maximize=True)
        >>> ind_A = Individual([0, 0, 1, 0, 1], IdentityDecoder, problem=f)
        >>> ind_A.fitness = 2
        >>> ind_B = Individual([1, 1, 1, 1, 1], IdentityDecoder, problem=f)
        >>> ind_B.fitness = 5
        >>> ind_A > ind_B
        False


        Use care when writing selection operators! When comparing `Individuals`, `>` always means "better than."
        The `>` function may indicate maximization, minimization, Pareto dominance, etc.: it all depends on the
        underlying `Problem`.

        >>> f = binary_problems.MaxOnes(maximize=False)
        >>> ind_A = Individual([0, 0, 1, 0, 1], IdentityDecoder, problem=f)
        >>> ind_A.fitness = 2
        >>> ind_B = Individual([1, 1, 1, 1, 1], IdentityDecoder, problem=f)
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
        return self.genome.__str__()

    def __repr__(self):
        return f"{type(self).__name__}({self.genome.__repr__()}, {self.decoder.__repr__()}, {self.problem.__repr__()})"


##############################
# Abstract Base Class Decoder
##############################
class Decoder(abc.ABC):

    @abc.abstractmethod
    def decode(self, genome):
        pass


##############################
# Class IdentityDecoder
##############################
class IdentityDecoder(Decoder):
    """A decoder that maps a genome to itself.  This acts as a 'direct' or 'phenotypic' encoding:
    Use this when your genotype and phenotype are the same thing."""

    def decode(self, genome):
        """:return: the input `genome`.

        For example:

        >>> d = IdentityDecoder()
        >>> d.decode([0.5, 0.6, 0.7])
        [0.5, 0.6, 0.7]
        """
        return genome

    def __repr__(self):
        return type(self).__name__ + "()"


##############################
# Class BinaryToIntDecoder
##############################
class BinaryToIntDecoder(Decoder):
    """A decoder that converts a Boolean-vector genome into an integer-vector phenome."""

    def __init__(self, *segments):
        """Constructs a decoder that will convert a binary representation into a corresponding
            int-value vector.

        :param segments: is a sequence of integer that determine how the binary sequence is to be
                         broken up into chunks for interpretation
        :return: a function for real-value phenome decoding of a sequence of binary digits

        The `segments` parameter indicates the number of (genome) bits per (phenome) dimension.  For example, if we
        construct the decoder

        >>> d = BinaryToIntDecoder(4, 3)

        then it will look for a genome of length 7, with the first 4 bits mapped to the first phenotypic value, and the
        last 3 bits making up the second:

        >>> d.decode([0,0,0,0,1,1,1])
        [0, 7]
        """
        self.segments = segments

    def decode(self, genome):
        """
        Converts a Boolean genome to an integer-vector phenome by interpreting each segment of the genome as
        low-endian binary number.

        :param genome: a list of 0s and 1s representing a Boolean genome
        :return: a corresponding list of ints representing the integer-vector phenome

        For example, a Boolean representation of [1, 12, 5] can be decoded like this:

        >>> d = BinaryToIntDecoder(4, 4, 4)
        >>> d.decode([0,0,0,1, 1, 1, 0, 0, 0, 1, 1, 0])
        [1, 12, 6]
        """

        # TODO the laborious string conversion approach could be replaced with something more elegant;
        # but this was a copy-n-paste job from some of my code from elsewhere that I knew worked.

        values = []
        offset = 0  # how far are we into the binary sequence

        for segment in self.segments:
            # snip out the next sequence
            cur_sequence = genome[offset:offset + segment]
            values.append(BinaryToIntDecoder.__binary_to_int(cur_sequence))
            offset += segment

        return values

    @staticmethod
    def __binary_to_int(b):
        """Convert the given binary string to the equivalent

        >>> BinaryToIntDecoder._BinaryToIntDecoder__binary_to_int([0, 1, 0, 1])
        5
        """
        return int(BinaryToIntDecoder.__binary_to_str(b), 2)

    @staticmethod
    def __binary_to_str(b):
        """Convert a vector of binary values into a simple string of binary.

        For example,

        >>> BinaryToIntDecoder._BinaryToIntDecoder__binary_to_str([0,1,0,1])
        '0101'
        """
        return "".join([str(x) for x in b])


##############################
# Class BinaryToRealDecoder
##############################
class BinaryToRealDecoder(Decoder):
    def __init__(self, *segments):
        """ This returns a function that will convert a binary representation into a corresponding
            real-value vector.  The segments are a collection of tuples that indicate how many bits
            per segment, and the corresponding real-value bounds for that segment.

        :param segments: is a sequence of tuples of the form (number of bits, minimum, maximum) values
        :return: a function for real-value phenome decoding of a sequence of binary digits

        For example, if we construct the decoder

        >>> d = BinaryToRealDecoder((4, -5.12, 5.12),(4, -5.12, 5.12))

        then it will look for a genome of length 8, with the first 4 bits mapped to the first phenotypic value, and the
        last 4 bits making up the second.  The traits have a minimum value of -5.12 (corresponding to 0000) and a
        maximum of 5.12 (corresponding to 1111):

        >>> d.decode([0, 0, 0, 0, 1, 1, 1, 1])
        [-5.12, 5.12]
        """
        # first we want to create an _int_ encoder since we'll be using that to do the first pass
        len_segments = list(pluck(0, segments))  # snip out just the binary segment lengths from the set of tuples

        cardinalities = [2 ** i for i in len_segments]  # how many possible values per segment

        # We will use this function to first decode to integers
        self.binary_to_int_decoder = BinaryToIntDecoder(*len_segments)

        # Now get the corresponding real value ranges
        self.lower_bounds = list(pluck(1, segments))
        self.upper_bounds = list(pluck(2, segments))

        # This corresponds to the amount each binary value is multiplied by to get the final real value (plus the lower
        # bound offset, of course)
        self.increments = [(upper - lower) / (cardinalities - 1) for lower, upper, cardinalities in
                           zip(self.lower_bounds, self.upper_bounds, cardinalities)]

    def decode(self, genome):
        int_values = self.binary_to_int_decoder.decode(genome)
        values = [l + i * inc for l, i, inc in zip(self.lower_bounds, int_values, self.increments)]
        return values


if __name__ == '__main__':
    pass
