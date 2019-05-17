#!/usr/bin/env python3
"""
    Functions and classes for decoding problem representations from genotypic to phenotypic spaces.

    These are optional utility classes.  Problem implementations are free to directly evaluate
    individuals if that is clearer.

    TODO Can add gray encoding as a decode function decorator, which would be pretty cool.

"""
import random

from toolz.itertoolz import pluck
from abc import ABCMeta, abstractmethod

import smallLEAP.reproduction


def binary_to_str(b):
    """ convert a vector of binary values into a simple string of binary

    :param b: vector of binary values, e.g., [0,1,0,1]
    :return: "0101"
    """
    return "".join([str(x) for x in b])


def binary_to_int(b):
    """ Convert the given binary string to the equivalent

    :param b:
    :return: the binary string converted to integer
    """
    return int(binary_to_str(b), 2)


def binary_to_int_encoding(*segments):
    """ This returns a function that will convert a binary representation into a corresponding
        int-value vector.

    E.g., binary_to_int_encode(4, 4) will return a function, f(), that can then be used this way:
    f([0,0,0,0,1,1,1,1]) returns [0,15]

    TODO the laborious string conversion approach could be replaced with something more elegant;
    but this was a copy-n-paste job from some of my code from elsewhere that I knew worked.

    :param segments: is a sequence of integer that determine how the binary sequence is to be
                     broken up into chunks for interpretation
    :return: a function for real-value phenome decoding of a sequence of binary digits
    """
    local_segments = segments

    def do_encoding(binary_sequence):
        nonlocal local_segments

        values = []
        offset = 0  # how far are we into the binary sequence

        for segment in local_segments:
            # snip out the next sequence
            cur_sequence = binary_sequence[offset:offset + segment]

            values.append(binary_to_int(cur_sequence))

            offset += segment

        return values

    return do_encoding


def binary_to_real_encoding(*segments):
    """ This returns a function that will convert a binary representation into a corresponding
        real-value vector.  The segments are a collection of tuples that indicate how many bits
        per segment, and the corresponding real-value bounds for that segment.

    E.g., binary_to_real_encode((4, -5.12, 5.12),(4, -5.12, 5.12)) will return a function, f(), that can then be
    used this way:

        f([0,0,0,0,1,1,1,1]) returns [-5.12,5.12]

    :param segments: is a sequence of tuples of the form (number of bits, minimum, maximum) values
    :return: a function for real-value phenome decoding of a sequence of binary digits
    """
    # first we want to create an _int_ encoder since we'll be using that to do the first pass
    len_segments = list(pluck(0, segments))  # snip out just the binary segment lengths from the set of tuples

    cardinalities = [2 ** i for i in len_segments]  # how many possible values per segment

    # We will use this function to first decode to integers
    local_binary_to_int_decoder = binary_to_int_encoding(*len_segments)

    # Now get the corresponding real value ranges
    lower_bounds = list(pluck(1, segments))
    upper_bounds = list(pluck(2, segments))

    # This corresponds to the amount each binary value is multiplied by to get the final real value (plus the lower
    # bound offset, of course)
    increments = [(upper - lower) / (cardinalities - 1) for lower, upper, cardinalities in
                  zip(lower_bounds, upper_bounds, cardinalities)]

    def do_encoding(binary_sequence):
        nonlocal lower_bounds
        nonlocal increments
        nonlocal local_binary_to_int_decoder

        int_values = local_binary_to_int_decoder(binary_sequence)

        values = [l + i * inc for l, i, inc in zip(lower_bounds, int_values, increments)]

        return values

    return do_encoding


class Encoding(metaclass=ABCMeta):
    """
        This encapsulates the value representation for individuals, thus abstracting away that the
        underlying representation may be genotypic or phenotypic.  After all, the associated
        Individual and its Problem only care about the ultimate value and not how that value is
        actually represented.

        TODO Was it really worthwhile to make self.sequence a property?

        self.sequence represents the raw representation for individuals; it is up to the subclasses to implement
        proper decoding functionality as well as means for randomly generating sequences valid for a given user context.

        TODO Should we force inclusion of a specific `decoder` member?
    """

    def __init__(self):
        # The actual sequence to be decode()'d.  This is representation specific. E.g., this could be a binary sequence
        # or a real-value vector or a list of lists of binary values or whatever.
        self._sequence = None

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, value):
        """
        :param value: to set the sequence to
        :return: None
        """
        # TODO We can add some sanity checking for valid sequences
        self._sequence = value


    @abstractmethod
    def decode(self):
        """
        :return: a decoded self._sequence
        """
        raise NotImplementedError

    @abstractmethod
    def random(self):
        """
        Create an appropriately randomly generated self._sequence

        :return: None
        """
        raise NotImplementedError


class StaticRealValueEncoding(Encoding):
    """
    A test-bed common real-value representation using a static std for a fixed length vector of real values

    """
    def __init__(self, length, lower_bound, upper_bound):
        """
        :param length: how many real-values in individual
        :param lower_bound: is the lower bound for initial random values
        :param upper_bound: is the upper bound for initial random values
        """
        super().__init__()

        self.length = length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self._sequence = None

    def decode(self):
        """ This is a phenotypic representation, so no decoding necessary.  Just return the vector of real-values.

        :return: sequence of real values
        """
        return self._sequence

    def random(self):
        """ Generate a random sequence in [lower,upper] for self.length real-value elements

        :return: None
        """
        self._sequence = smallLEAP.reproduction.create_real_value_sequence(self.length, self.lower_bound, self.upper_bound)




class BinaryEncoding(Encoding):
    """
    ABC for common binary encoding representations
    """

    def __init__(self, decoder=None):
        """
        self.total_bits need to be updated in subclasses since that's used to properly create random individuals.

        TODO Consider moving self.decoder to Encoding?

        :param decoder: is the decoding function used to translate the bits
        """
        super().__init__()

        self.decoder = decoder
        self.total_bits = None  # this needs to be calculated in subclasses

    def decode(self):
        """ Depends on an decoder being defined

        :return: decoded binary sequence
        """
        return self.decoder(self.sequence)

    def random(self):
        """ Stochastically generates self.sequence to self.total_bits number of bits.

        :return: None
        """
        self.sequence = smallLEAP.reproduction.create_binary_sequence(self.total_bits)

    def __str__(self):
        return binary_to_str(self.sequence) + ' -> ' + str(self.decode())


class SimpleBinaryEncoding(BinaryEncoding):
    """
        This is for problems where we do want to directly consider binary sequences without translation.  E.g., for
        things like the MAX ONES problem.
    """

    def __init__(self, total_bits):
        """
        Note that we *intentionally* do not pass in a decoder since we're always going to return the raw binary
        string when decoding.

        :param total_bits: for each individual
        """
        super().__init__()

        self.total_bits = total_bits

    def decode(self):
        """

        :return: raw binary sequence
        """
        return self.sequence


class BinaryIntValueEncoding(BinaryEncoding):
    """
        Simple binary-value genotypic encoding for a sequence of integers.

        The integer sequence has a corresponding series of binary segments.  The ctor is used
        to define the number of such segments and their respective lengths.
    """

    def __init__(self, *segments):
        """

        :param segments: is a sequence of integers specifying the size of consecutive bit segments
        """
        super().__init__(decoder=binary_to_int_encoding(*segments))

        # We need to remember the total number of bits to ensure we can properly randomly generate new sequences
        self.total_bits = sum(segments)


class BinaryRealValueEncoding(BinaryEncoding):
    """
        Simple binary-value genotypic encoding for a sequence of real values.

        The real-value sequence has a corresponding series of binary segments.  The ctor is used
        to define the number of such segments and their respective lengths as well as bounds.
    """

    def __init__(self, *segments):
        """ E.g., (4, -5.12, 5.12), (7, 22, 150) specifies eleven total bits with the first four bits corresponding
        to real-values in [-5.12,5.12] and the remaining 7 in [22,150]

        :param segments: is a sequence of tuples describing real values and their bounds of the
                         form (bits, lower, upper)
        """
        super().__init__(decoder=binary_to_real_encoding(*segments))

        # We need to remember the total number of bits to ensure we can properly randomly generate new sequences
        self.total_bits = sum(list(pluck(0, segments)))


if __name__ == '__main__':
    # TODO These need to be moved to proper unit tests

    # First, binary encoding/decoding tests
    my_binary_encoder = binary_to_int_encoding(3, 3)

    values = my_binary_encoder([0, 0, 0, 1, 1, 1])

    print(values)

    # Then the real-value encoding/decoding tests using four genes of three bits to represent [-5.12,5.12]
    my_real_value_decoder = binary_to_real_encoding(
        (3, -5.12, 5.12), (3, -5.12, 5.12), (3, -5.12, 5.12), (3, -5.12, 5.12))

    values = my_real_value_decoder([0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])

    print(values)

    # Now to exercise new Encoding classes
    my_binary_encoding = BinaryIntValueEncoding(3, 3, 3)
    my_binary_encoding.random()
    print(my_binary_encoding)

    my_binary_encoding = BinaryRealValueEncoding((3, -5.12, 5.12), (3, -5.12, 5.12))
    my_binary_encoding.random()
    print(my_binary_encoding)

    my_binary_encoding = SimpleBinaryEncoding(5)
    my_binary_encoding.random()
    print(my_binary_encoding)

    # Real-value tests
    my_real_value_encoding = StaticRealValueEncoding(5, 1, 5)
    my_real_value_encoding.random()
    print(my_real_value_encoding.sequence)

