#!/usr/bin/env python3
"""
    Used to initialize segments
"""

##############################
# Closure create_segmented_sequence
##############################
def create_segmented_sequence(length, seq_initializer):
    """ Create a segmented test_sequence

    A segment is a list of lists.  `seq_initializer` is used to create `length`
    individual segments, which allows for the using any of the pre-supplied
    initializers for a regular genomic test_sequence, or for making your own.

    `length` denotes how many segments to generate.  If it's an integer, then
    we will create `length` segments.  However, if it's a function that draws
    from a random distribution that returns an int, we will, instead, use that
    to calculate the number of segments to generate.

    >>> from leap_ec.binary_rep.initializers import create_binary_sequence
    >>> segments = create_segmented_sequence(3, create_binary_sequence(3))
    >>> assert len(segments) == 3


    :param length: How many segments?
    :type length: int or Callable
    :param seq_initializer: initializer for creating individual sequences
    :type seq_initializer: Callable
    :return: test_sequence of segments
    :rtype: list
    """
    if callable(length):
        num_segments = length()
    else:
        num_segments = length

    segments = [seq_initializer() for _ in range(num_segments)]

    return segments

