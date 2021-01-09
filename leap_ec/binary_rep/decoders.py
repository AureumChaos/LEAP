#!/usr/bin/env python3
"""
    Decoders for binary representations.
"""
from toolz.itertoolz import pluck

from .. decoder import Decoder

##############################
# Class BinaryToIntDecoder
##############################
class BinaryToIntDecoder(Decoder):
    """A decoder that converts a Boolean-vector genome into an integer-vector
    phenome. """

    def __init__(self, *descriptors):
        """Constructs a decoder that will convert a binary representation
        into a corresponding int-value vector.

        :param descriptors: is a test_sequence of integer that determine how the
            binary test_sequence is to be broken up into chunks for interpretation

        :return: a function for real-value phenome decoding of a test_sequence of
            binary digits

        The `segments` parameter indicates the number of (genome) bits per (
        phenome) dimension.  For example, if we construct the decoder

        >>> d = BinaryToIntDecoder(4, 3)

        then it will look for a genome of length 7, with the first 4 bits
        mapped to the first phenotypic value, and the last 3 bits making up
        the second:

        >>> d.decode([0,0,0,0,1,1,1])
        [0, 7]
        """
        super().__init__()
        self.descriptors = descriptors

    def decode(self, genome, *args, **kwargs):
        """
        Converts a Boolean genome to an integer-vector phenome by
        interpreting each segment of the genome as low-endian binary number.

        :param genome: a list of 0s and 1s representing a Boolean genome

        :return: a corresponding list of ints representing the integer-vector
            phenome

        For example, a Boolean representation of [1, 12, 5] can be decoded
        like this:

        >>> d = BinaryToIntDecoder(4, 4, 4)
        >>> d.decode([0,0,0,1, 1, 1, 0, 0, 0, 1, 1, 0])
        [1, 12, 6]
        """

        # TODO the laborious string conversion approach could be replaced
        #  with something more elegant; but this was a copy-n-paste job from
        #  some of my code from elsewhere that I knew worked.

        values = []
        offset = 0  # how far are we into the binary test_sequence

        for descriptor in self.descriptors:
            # snip out the next test_sequence
            cur_sequence = genome[offset:offset + descriptor]
            values.append(BinaryToIntDecoder.__binary_to_int(cur_sequence))
            offset += descriptor

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
# Class BinaryToRealDecoderCommon
##############################
class BinaryToRealDecoderCommon(Decoder):
    """
        Common implementation for binary to real decoders.

        The base classes BinaryToRealDecoder and BinaryToRealGreyDecoder differ
        by just the underlying binary to integer decoder.  Most all the rest
        of the binary integer to real-value decoding is the same, hence this
        class.
    """

    def __init__(self, *segments):
        """
        :param segments: is a test_sequence of tuples of the form (number of bits,
            minimum, maximum) values

        :return: a function for real-value phenome decoding of a test_sequence of
            binary digits
        """
        super().__init__()

        # Verify that segments have the correct dimensionality
        for i, seg in enumerate(segments):
            if len(seg) != 3:
                raise ValueError("Each segment must be a have exactly three "
                                 "elements (num_bits, min, max), " +
                                 f"but segment {i} is '{seg}'.'")

        # first we want to create an _int_ encoder since we'll be using that
        # to do the first pass

        # snip out just the binary segment lengths from the set of tuples;
        # we save this for the subclasses for their binary to integer decoders
        self.len_segments = list(pluck(0, segments))

        # how many possible values per segment
        cardinalities = [2 ** i for i in self.len_segments]

        # We will use this function to first decode to integers.
        # This is assigned in the sub-classes depending on whether we want to
        # use grey encoding or not to convert from binary to integer sequences.
        self.binary_to_int_decoder = None

        # Now get the corresponding real value ranges
        self.lower_bounds = list(pluck(1, segments))
        self.upper_bounds = list(pluck(2, segments))

        # This corresponds to the amount each binary value is multiplied by
        # to get the final real value (plus the lower bound offset, of course)
        self.increments = [(upper - lower) / (cardinalities - 1) for
                           lower, upper, cardinalities in
                           zip(self.lower_bounds, self.upper_bounds,
                               cardinalities)]

    def decode(self, genome, *args, **kwargs):
        """Convert a list of binary values into a real-valued vector."""
        int_values = self.binary_to_int_decoder.decode(genome)
        values = [l + i * inc for l, i, inc in
                  zip(self.lower_bounds, int_values, self.increments)]
        return values


##############################
# Class BinaryToRealDecoder
##############################
class BinaryToRealDecoder(BinaryToRealDecoderCommon):
    def __init__(self, *segments):
        """ This returns a function that will convert a binary representation
        into a corresponding real-value vector.  The segments are a
        collection of tuples that indicate how many bits per segment, and the
        corresponding real-value bounds for that segment.

        :param segments: is a test_sequence of tuples of the form (number of bits,
            minimum, maximum) values

        :return: a function for real-value phenome decoding of a test_sequence of
            binary digits

        For example, if we construct the decoder
        then it will look for a genome of length 8, with the first 4 bits
        mapped to the first phenotypic value, and the last 4 bits making up
        the second.  The traits have a minimum value of -5.12 (corresponding
        to 0000) and a maximum of 5.12 (corresponding to 1111):

        >>> d = BinaryToRealDecoder((4, -5.12, 5.12),(4, -5.12, 5.12))
        >>> d.decode([0, 0, 0, 0, 1, 1, 1, 1])
        [-5.12, 5.12]
        """
        super().__init__(*segments)

        # We will use this function to first decode to integers
        self.binary_to_int_decoder = BinaryToIntDecoder(*self.len_segments)


##############################
# Class BinaryToIntGreyDecoder
##############################
class BinaryToIntGreyDecoder(BinaryToIntDecoder):
    """ This performs Gray encoding when converting from binary strings.

        See also:
        https://en.wikipedia.org/wiki/Gray_code#Converting_to_and_from_Gray_code

        For example, a grey encoded Boolean representation of [1, 8, 4] can
        be decoded like this:

        >>> d = BinaryToIntGreyDecoder(4, 4, 4)
        >>> d.decode([0,0,0,1, 1, 1, 0, 0, 0, 1, 1, 0])
        [1, 8, 4]
    """

    def __init__(self, *descriptors):
        super().__init__(*descriptors)

    @staticmethod
    def __gray_encode(num):
        """
        https://en.wikipedia.org/wiki/Gray_code#Converting_to_and_from_Gray_code

        :param value: integer value to be gray encoded
        :return: gray encoded integer
        """
        mask = num >> 1

        while mask != 0:
            num = num ^ mask
            mask = mask >> 1

        return num

    def decode(self, genome, *args, **kwargs):
        # First decode the integers from the binary representation using
        # regular binary decoding.
        values = super().decode(genome)

        gray_encoded_values = [BinaryToIntGreyDecoder.__gray_encode(v) for v in
                               values]

        return gray_encoded_values


##############################
# Class BinaryToRealGreyDecoder
##############################
class BinaryToRealGreyDecoder(BinaryToRealDecoderCommon):
    def __init__(self, *segments):
        """ This returns a function that will convert a binary representation
        into a corresponding real-value vector.  The segments are a
        collection of tuples that indicate how many bits per segment, and the
        corresponding real-value bounds for that segment.

        :param segments: is a test_sequence of tuples of the form (number of bits,
            minimum, maximum) values :return: a function for real-value phenome
            decoding of a test_sequence of binary digits

        For example, if we construct the decoder then it will look for
        a genome of length 8, with the first 4 bits mapped to the first
        phenotypic value, and the last 4 bits making up the second.  The
        traits have a minimum value of -5.12 (corresponding to 0000) and a
        maximum of 5.12 (corresponding to 1111):

        >>> d = BinaryToRealGreyDecoder((4, -5.12, 5.12),(4, -5.12, 5.12))
        >>> d.decode([0, 0, 0, 0, 1, 1, 1, 1])
        [-5.12, 1.706666666666666]
        """
        super().__init__(*segments)

        # We will use this function to first decode to integers
        self.binary_to_int_decoder = BinaryToIntGreyDecoder(*self.len_segments)


if __name__ == '__main__':
    pass
