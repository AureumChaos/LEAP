"""
    A set of standard EA problems that rely on a binary-representation
"""
from itertools import groupby

import numpy as np
from PIL import Image, ImageOps

from leap_ec.problem import ScalarProblem


##############################
# Class MaxOnes
##############################
class MaxOnes(ScalarProblem):
    """
    Implementation of the classic max-ones problem, where the individuals
    are represented by a bit vector.

    By default, the number of 1's in the phenome are maximized.
    
    >>> p = MaxOnes()

    But an optional target string can also be specified, in which case the 
    number of matches to the target are maximized:

    >>> import numpy as np
    >>> p = MaxOnes(target_string=np.array([1, 1, 1, 1, 1, 0, 0, 0 ,0]))
    """

    def __init__(self, target_string=None, maximize=True):
        super().__init__(maximize)
        self.target_string = target_string

    def evaluate(self, individual):
        """
        By default this counts the number of 1's:

        >>> from leap_ec.individual import Individual
        >>> import numpy as np
        >>> p = MaxOnes()
        >>> ind = Individual(np.array([0, 0, 1, 1, 0, 1, 0, 1, 1]),
        ...                   problem=p)
        >>> p.evaluate(ind)
        5

        Or, if a target string was given, we count matches:

        >>> from leap_ec.individual import Individual
        >>> import numpy as np
        >>> p = MaxOnes(target_string=np.array([1, 1, 1, 1, 1, 0, 0, 0 ,0]))
        >>> ind = Individual(np.array([0, 0, 1, 1, 0, 1, 0, 1, 1]),
        ...                   problem=p)
        >>> p.evaluate(ind)
        3
        """
        if not isinstance(individual.phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(individual.phenome)}."))
        # If we've 
        if self.target_string is not None:
            assert(len(individual.phenome) == len(self.target_string)), f"Fitness function target string has {len(self.target_string)} dimensions, but received a phenome with {len(individual.phenome)} dimensions."
            return np.sum(individual.phenome == self.target_string)
        else:
            return np.count_nonzero(individual.phenome == 1)


##############################
# Class LeadingOnes
##############################
class LeadingOnes(ScalarProblem):
    """
    Implementation of the classic leading-ones problem, where the individuals
    are represented by a bit vector.

    By default, the number of consecutve 1's starting from the beginning of
    the phenome are maximized:
    
    >>> p = LeadingOnes()

    But an optional target string can also be specified, in which case the 
    number of matches to the target are maximized:

    >>> import numpy as np
    >>> p = LeadingOnes(target_string=np.array([1, 1, 0, 1, 1, 0, 0, 0 ,0]))
    """

    def __init__(self, target_string=None, maximize=True):
        super().__init__(maximize)
        self.target_string = target_string

    def evaluate(self, individual):
        """
        By default this counts the number of consecutive 1's at the
        start of the string:

        >>> from leap_ec.individual import Individual
        >>> import numpy as np
        >>> p = LeadingOnes()
        >>> ind = Individual(np.array([1, 1, 1, 1, 0, 1, 0, 1, 1]),
        ...                   problem=p)
        >>> p.evaluate(ind)
        4

        Or, if a target string was given, we count matches:

        >>> from leap_ec.individual import Individual
        >>> import numpy as np
        >>> p = LeadingOnes(target_string=np.array([1, 1, 0, 1, 1, 0, 0, 0 ,0]))
        >>> ind = Individual(np.array([1, 1, 1, 1, 0, 1, 0, 1, 1]),
        ...                   problem=p)
        >>> p.evaluate(ind)
        2
        """
        if not isinstance(individual.phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(individual.phenome)}."))
        
        if self.target_string is not None:
            assert(len(individual.phenome) == len(self.target_string)), f"Fitness function target string has {len(self.target_string)} dimensions, but received a phenome with {len(individual.phenome)} dimensions."
            match_str = (individual.phenome == self.target_string)
        else:
            match_str = (individual.phenome == 1)

        groups = groupby(match_str)
        _, leading_vals = next(groups)
        return np.sum(list(leading_vals))


##############################
# Class DeceptiveTrap
##############################
class DeceptiveTrap(ScalarProblem):
    """
    A simple bi-modal function whose global optimum is the Boolean vector
    of all 1's, but in which fitness *decreases* as the number of 1's in
    the vector *increases*—giving it a local optimum of [0, ..., 0] with a 
    very wide basin of attraction.
    """
    def __init__(self, maximize=True):
        super().__init__(maximize=maximize)
    
    def evaluate(self, individual):
        
        """
        >>> from leap_ec.individual import Individual
        >>> import numpy as np
        >>> p = DeceptiveTrap()

        The trap function has a global maximum when the number of one's
        is maximized:

        >>> ind = Individual(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        >>> p.evaluate(ind)
        10

        It's minimized when we have just one zero:
        >>> ind = Individual(np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1]))
        >>> p.evaluate(ind)
        0

        And has a local optimum when we have no ones at all:
        >>> ind = Individual(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        >>> p.evaluate(ind)
        9
        """
        dimensions = len(individual.phenome)
        max_ones = np.count_nonzero(individual.phenome == 1)
        if max_ones == dimensions:
            return dimensions
        else:
            return dimensions - max_ones - 1


##############################
# Class TwoMax
##############################
class TwoMax(ScalarProblem):
    """
    A simple bi-modal function that returns the number of 1's if there
    are more 1's than 0's, else the number of 0's.

    Also known as the "Twin-Peaks" problem.
    """
    def __init__(self, maximize=True):
        super().__init__(maximize=maximize)
    
    def evaluate(self, individual):
        
        """
        >>> from leap_ec.individual import Individual
        >>> import numpy as np
        >>> p = TwoMax()

        The TwoMax problems returns the number over 1's if they are
        in the majority:

        >>> ind = Individual(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]))
        >>> p.evaluate(ind)
        7

        Else the number of zeros:
        >>> ind = Individual(np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1]))
        >>> p.evaluate(ind)
        6
        """
        dimensions = len(individual.phenome)
        max_ones = np.count_nonzero(individual.phenome == 1)
        return int(np.abs(dimensions/2 - max_ones) + dimensions/2)


##############################
# Class ImageProblem
##############################
class ImageProblem(ScalarProblem):
    """A variation on `max_ones` that uses an external image file to define a
    binary target pattern. """

    def __init__(self, path, maximize=True, size=(100, 100)):
        super().__init__(maximize)
        self.size = size
        self.img = ImageProblem._process_image(path, size)
        self.flat_img = np.ndarray.flatten(np.array(self.img))

    @staticmethod
    def _process_image(path, size):
        """Load an image and convert it to black-and-white."""
        x = Image.open(path)
        x = ImageOps.fit(x, size)
        return x.convert('1')

    def evaluate(self, individual):
        assert (len(individual.phenome) == len(self.flat_img)
                ), f"Bad genome length: got {len(individual.phenome)}, expected " \
                   f"{len(self.flat_img)} "
        diff = np.logical_not(individual.phenome ^ self.flat_img)
        return sum(diff)
