#!/usr/bin/env python3
""" This is for "segmented" encodings that are essentially sequences of
sequences.  This is useful for Pitt Approach implementations, as well as
other EAs that rely on similar representations.
"""
from copy import deepcopy
import random
from abc import ABCMeta, abstractmethod
import itertools
import functools
import toolz
import logging

from smallLEAP.encoding import Encoding, SimpleBinaryEncoding, BinaryIntValueEncoding
from smallLEAP.reproduction import binary_flip_mutation, clone_generator
import smallLEAP.individual
import smallLEAP.probes

class Segmented(Encoding, metaclass=ABCMeta):
    """
    This Encoding is an ABC that supports "sequences of sequences"
    representations.  This relies on other a sequence of decoders to
    represent the "sub-sequences".
    """

    def __init__(self, decoder_seq):
        """
        :param decoder_seq: is an Encoding representing the sub-sequences, and will be cloned for new instances
        """
        super().__init__()

        # All SegmentedEncodings will start with an empty sequence to which
        # sub-sequences are later added.
        self.sequence = []

        # This is an implementation Prototype pattern whereby this will be
        # cloned for self.random() calls.
        self.decoder_seq = decoder_seq

    @abstractmethod
    def _max_rand_segments(self):
        """ Over-ridden by sub-classes to control the number of randomly generated segments

        :return: an upper bound for the number of randomly generated segments
        """
        raise NotImplementedError

    def decode(self):
        """ This just naively decodes each encoding segment. You may want to over-ride this
            method if you want more sophisticated decoding/interpretation of segments.

        :return: decoded segments
        """
        return [x.decode() for x in self.sequence]

    def random(self):
        """ Randomly generate self._max_rand_segments()

        :return: None
        """
        for i in range(self._max_rand_segments()):
            # Since we're using a Prototype design pattern, clone the prototypical encoding
            # and add it as a new sequence.
            new_encoding = deepcopy(self.sub_encoding)
            new_encoding.random()
            self.sequence.append(new_encoding)

    def __str__(self):
        """
        :return: Return the str() print for each segment
        """
        return str([str(x) for x in self.sequence])


class FixedSegmented(Segmented):
    """
        These are for fixed number of segments per individual
    """

    def __init__(self, decoder_seq, length):
        """
        :param decoder_seq: is an Encoding representing the sub-sequences
        :param length: how many segments per individual
        """
        super().__init__(decoder_seq)

        self.length = length

    def _max_rand_segments(self):
        """ We want all randomly generated segments to be the same length

        :return: value pre-defined for number of segments
        """
        return self.length


class VaryingSegmented(Segmented):
    """
        For a variable number of segments per individual.
    """

    def __init__(self, decoder_seq, max_length):
        """
        TODO have max_length become an integer for uniform distribution of [1,n], or an optional
        function that returns an integer for exact number of segments to be created for a given
        call to self.random() where the integer is drawn from a distribution implemented in a
        user-defined function.

        :param decoder_seq: is an Encoding representing the sub-sequences
        :param max_length: is the maximum number of randomly uniformly generated sub-sequences
        """
        super().__init__(decoder_seq)

        self.max_length = max_length

    def _max_rand_segments(self):
        return random.randint(1, self.max_length)

def segmented_binary_flip_mutation(individual, expected_per_individual, expected_per_segment):
    """
        Presumes a binary representation in segmented individual.sequence, and that the sub-sequence encodings are
        of the same length.

        :param individual: is individual to possibly be changed
        :param expected_per_individual: is the expected number of mutated segments per individual
        :param expected_per_segment: is the expected number of bits to be flipped per selected segment

        :return: a copy of individual with individual.sequence bits flipped based on probability
    """
    # This is the likelihood of mutating a given segment
    probability_per_segment = expected_per_individual * 1.0 / len(individual.encoding.sequence)

    # There is a chance that the expected number of segments to be mutated exceed the current number of segments,
    # in which case we just cap that at 100%.
    probability_per_segment = min(probability_per_segment, 1.0)

    # This is the probability of flipping an individual bit within a segment
    probability_per_bit= expected_per_segment * 1.0 / len(individual.encoding.sequence[0].sequence)

    def flip(bit):
        # TODO this is duplicated in reproduction.py, so should be consolidated
        if random.random() < probability_per_bit:
            return (bit + 1) % 2
        else:
            return bit

    def bit_flip_subsequence(sequence):
        return [flip(bit) for bit in sequence]

    def bit_flip_encodings(encoding):
        if random.random() < probability_per_segment:
            encoding.sequence = bit_flip_subsequence(encoding.sequence)

        return encoding

    individual.encoding.sequence = [bit_flip_encodings(sub_sequence) for sub_sequence in individual.encoding.sequence]

    return individual


def segmented_binary_flip_mutation_generator(next_individual, expected_per_individual, expected_per_segment):
    """ Generator for mutating an individual and passing it down the pipe

    :param next_individual: where to get the next individual
    :param expected_per_individual:  is the expected number of mutated segments per individual
    :param expected_per_segment: is the expected number of bits to be flipped per selected segment
    :return: yields a mutated segmented individual
    """
    while True:
        yield segmented_binary_flip_mutation(next(next_individual), expected_per_individual, expected_per_segment)


def segmented_add_mutation(individual, probability):
    """ This adds segments with a certain probability.

    This will add at most one segment at a random location.

    TODO May want to reconsider only allowing up to one segment to be added.

    FIXME Need to change first to roll the dice if we're going to even add a segment, then randomly choose the
    position. Currently, if we select, say, 100%, then this will *always* prepend a new segment when we'd prefer
    that it be uniformly randomly placed.

    :param individual: to whom we may add a new segment
    :return: individual with possible new segement
    """

    if random.random() < probability:
        # randomly determine *where* we're going to insert new segment
        # (randrange(len + 1) to allow for possibly appending to sequence)
        pos = random.randrange(len(individual.encoding.sequence) + 1)

        # Create new encoding object; we use deepcopy so that we can get the
        # important state such as what the encoding actually represents. This
        # is why we *don't* just naively instantiate __class__ to get a new
        # Encoding object because that would be a complete blank slate.
        # Arbitrarily pick the first sequence since they're all homogeneous.
        new_encoding = deepcopy(individual.encoding.sequence[0])

        # Randomize it
        new_encoding.random()

        # And then insert it
        individual.encoding.sequence.insert(pos, new_encoding)

    return individual


def segmented_add_mutation_generator(next_individual, probability):
    """ Generator for possibly adding segments to individuals with given
    probability

    :param next_individual: where to get the next individual
    :param probability: of adding a segment to the individual
    :return: yields individual with possible new segment
    """
    while True:
        yield segmented_add_mutation(next(next_individual), probability)


def segmented_remove_mutation(individual, probability):
    """ This removes segments with a certain probability.

    Individuals with single segments are ignored.  Will remove at most a *single* segment.

    TODO May want to reconsider only allowing up to one segment to be deleted.

    :param individual:
    :param probability:
    :return: individual with a segment possibly removed
    """
    if len(individual.encoding.sequence) == 1:
        return individual

    for i in range(len(individual.encoding.sequence)):
        if random.random() < probability:
            del individual.encoding.sequence[i]
            break

    return individual


def segmented_remove_mutation_generator(next_individual, probability):
    """ Generator for possibly removing segments from an individual with a given probability

    :param next_individual: from which to get the next individual
    :param probability: of removing a single segment
    :return: yields individual with a possible segment removed
    """
    while True:
        yield segmented_remove_mutation(next(next_individual), probability)


if __name__ == '__main__':
    # test fixed length segments
    my_fixed_segmented_encoding = FixedSegmented(SimpleBinaryEncoding(3), 3)
    my_fixed_segmented_encoding.random()
    print(my_fixed_segmented_encoding)

    # test varying length segments
    my_varying_segmented_encoding = VaryingSegmented(BinaryIntValueEncoding(3, 4, 3), 5)
    my_varying_segmented_encoding.random()
    print(my_varying_segmented_encoding)

    # test binary flip mutation
    #    Test individual has two segments of three bits
    ind = smallLEAP.individual.Individual(encoding=FixedSegmented(SimpleBinaryEncoding(3), 2))
    print('Original ind:', ind.encoding)

    # invert all bits to verify mutation worked
    new_ind = segmented_binary_flip_mutation(ind, 1, 1) # expect one mutated segment and one bit flipped
    print('Original ind after mutation:', ind.encoding)
    print('New ind:', new_ind.encoding)

    # now to the sames tests, but with varying length encodings

    ind = smallLEAP.individual.Individual(encoding=VaryingSegmented(BinaryIntValueEncoding(3, 4, 3), 5))
    print('Original varying:', ind.encoding)

    new_ind = segmented_binary_flip_mutation(ind, 1, 1) # expect about one mutated segment and one bit flipped
    print('Original varying ind after mutation:', ind.encoding)
    print('New varying ind:', new_ind.encoding)

    # Deletion testing
    del_ind = segmented_remove_mutation(new_ind, .1)
    print('Deleted segment:', del_ind.encoding)

    # Add testing
    added_ind = segmented_add_mutation(del_ind, .3333)
    print('Added:', added_ind.encoding)

    # Pipeline test -- now to exercise all the operators in a pipeline
    logging.basicConfig(level=logging.DEBUG)

    pop = [ind, new_ind, del_ind, added_ind]

    final_ind = toolz.pipe(itertools.cycle(pop), # cycle over the same individuals over and over, but we'll just pick the first guy for now
                           clone_generator,
                           functools.partial(smallLEAP.probes.log_individual, prefix='Before mutation: '),
                           functools.partial(segmented_binary_flip_mutation_generator, expected_per_individual=1, expected_per_segment=2),
                           functools.partial(smallLEAP.probes.log_individual, prefix='After mutation, before add: '),
                           functools.partial(segmented_add_mutation_generator, probability=.2),
                           functools.partial(smallLEAP.probes.log_individual, prefix='After add: '),
                           functools.partial(smallLEAP.reproduction.create_pool, size=1)
                           )

    print('Final ind:', [str(ind.encoding) for ind in final_ind])

    print('Done')

