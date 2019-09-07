from toolz.itertoolz import pluck

# TODO Add ABC.  Helps with docs, and adding new methods if we want to change it later.


##############################
# Class IdentityDecoder
##############################
class IdentityDecoder:
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


##############################
# Class BinaryToIntDecoder
##############################
class BinaryToIntDecoder:
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
class BinaryToRealDecoder:
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
