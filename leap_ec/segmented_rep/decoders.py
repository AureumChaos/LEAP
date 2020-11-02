#!/usr/bin/env python3
"""
    Used to decode segments
"""
from leap_ec.decoder import Decoder

##############################
# Class SegmentedDecoder
##############################
class SegmentedDecoder(Decoder):
    """
        For decoding LEAP segmented representations

        >>> from leap_ec.binary_rep.decoders import BinaryToIntDecoder

        This example presumes that each segment has five bits, the first to
        map to an integer and the remaining three to a different integer.

        >>> decoder = SegmentedDecoder(BinaryToIntDecoder(2,3))
        >>> genome = [[1, 0, 1, 0, 1], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1]]
        >>> vals = decoder.decode(genome)
        >>> assert vals == [[2, 5], [0, 7], [2, 1]]
    """
    def __init__(self, segment_decoder):
        """
        :param segment_decoder: is the decoder used for all segments
        :type segment_decoder:
        """
        super().__init__()

        self.segment_decoder = segment_decoder

    def decode(self, genome, *args, **kwargs):
        """
            For decoding `genome` which is a list of lists, or a
            segmented representation.

        :param genome: for a given individual
        :type genome: will be a list of segments (or lists)
        :param args: optional args
        :type args: list
        :param kwargs: optional keyword args
        :type kwargs: dict
        :return: a list of list of values decoded from `genome`
        :rtype: list
        """
        values = [self.segment_decoder.decode(segment) for segment in genome]

        return values
