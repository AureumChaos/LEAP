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

        :param genome:
        :type genome:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        values = [self.segment_decoder.decode(segment) for segment in genome]

        return values
