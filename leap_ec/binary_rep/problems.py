"""
    A set of standard EA problems that rely on a binary-representation
"""
import numpy as np
from PIL import Image, ImageOps

from leap_ec.problem import ScalarProblem


##############################
# Class MaxOnes
##############################
class MaxOnes(ScalarProblem):
    """
    Implementation of MAX ONES problem where the individuals are represented
    by a bit vector

    We don't need an encoder since the raw genome is *already* in the
    phenotypic space.
    """

    def __init__(self, maximize=True):
        """
        Create a MAX ONES problem with individuals that have bit vectors of
        size `length`
        """
        super().__init__(maximize)

    def evaluate(self, individual):
        """
        >>> from leap_ec.individual import Individual
        >>> import numpy as np
        >>> p = MaxOnes()
        >>> ind = Individual(np.array([0, 0, 1, 1, 0, 1, 0, 1, 1]),
        ...                   problem=p)
        >>> p.evaluate(ind)
        5
        """
        if not isinstance(individual.phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(individual.phenome)}."))
        return np.count_nonzero(individual.phenome == 1)


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
