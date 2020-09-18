""" This module provides machinery for representing executable objects
of various kinds—ex. functions, agent controllers, etc.—that serve as problem 
¯solutions.

A LEAP `Executable` is a kind of phenotype, and it is constructed when we use a
:class:`~leap_ec.core.Decoder` from the `executable.decoder` module to convert a 
genotypic representation of the object into an executable phenotype.
"""

import abc
from enum import Enum
import time

import numpy as np

from leap_ec.problem import ScalarProblem
from leap_ec.decoder import Decoder

##############################
# Abstract Class Executable
##############################
class Executable(abc.ABC):
    @abc.abstractmethod
    def output(self, input):
        pass


##############################
# Class RandomExecutable
##############################
class RandomExecutable(Executable):
    """
    A trivial `Executable` phenotype that samples a random value from its 
    output space.

    :param input_space: space of possible inputs (ignored)
    :param output_space: the space of possible actions to sample from,
        satisfying the `Space` interface used by OpenAI Gym
    """

    def __init__(self, input_space, output_space):
        assert (output_space is not None)
        assert (hasattr(output_space, 'sample'))
        self.input_space = input_space
        self.output_space = output_space

    def output(self, input):
        """
        Return a random output.

        :param input: ignored
        :return: a randomly selection action from the output space

        For example, if we use a space from OpenAI Gym that defines a 2-D box of continuous values:

        >>> from gym import spaces
        >>> import numpy as np
        >>> output_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float32)

        Then this method will sample a random 2-D point in that box:

        >>> b = RandomExecutable(None, output_space)
        >>> b.output(input='whatever')
        array([..., ...], dtype=float32)
        """
        return self.output_space.sample()


##############################
# Class KeyboardExecutable
##############################
class KeyboardExecutable(Executable):
    """
    A non-autonomous `Executable` phenotype that allows users to control an 
    agent via the keyboard.

    :param input_space: space of possible inputs (ignored)
    :param output_space: the space of possible actions to sample from,
        satisfying the `Space` interface used by OpenAI Gym
    :param keymap: `dict` mapping keys to elements of the output space
    """

    def __init__(self, input_space, output_space,
                 keymap=lambda x: int(x - ord('0')) if x in range(ord('0'), ord(
                     '9')) else 0):
        assert (output_space is not None)
        if np.prod(output_space.shape) > 1:
            raise ValueError(
                "This environment requires an Executable with {0} ".format(
                    np.prod(output_space.shape)) +
                "outputs, but {0} can only produce 1 output at a time.".format(
                    KeyboardExecutable.__name__))
        assert (keymap is not None)
        self.output_space = output_space
        self.keymap = keymap
        self.action = 0

    def key_press(self, key, mod):
        """You'll need to assign this function to your environment's
        key_press handler. """
        self.action = self.keymap(key)

    def key_release(self, key, mod):
        """You'll need to assign this function to your environment's
        key_release handler. """
        if self.keymap(key) == self.action:
            self.action = 0

    def output(self, input):
        time.sleep(0.05)
        return self.action
