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

    def evaluate(self, phenome):
        """
        >>> from leap_ec.individual import Individual
        >>> from leap_ec.decoder import IdentityDecoder
        >>> p = MaxOnes()
        >>> ind = Individual([0, 0, 1, 1, 0, 1, 0, 1, 1],
        ...                   decoder=IdentityDecoder(),
        ...                   problem=p)
        >>> p.evaluate(ind.decode())
        5
        """
        return phenome.count(1)


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

    def evaluate(self, phenome):
        assert (len(phenome) == len(self.flat_img)
                ), f"Bad genome length: got {len(phenome)}, expected " \
                   f"{len(self.flat_img)} "
        diff = np.logical_not(phenome ^ self.flat_img)
        return sum(diff)
