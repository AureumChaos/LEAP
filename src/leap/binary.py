from leap.problem import ScalarProblem
from leap import core


##############################
# Class MaxOnes
##############################
class MaxOnes(ScalarProblem):
    """
    Implementation of MAX ONES problem where the individuals are represented
    by a bit vector

    We don't need an encoder since the raw genome is *already* in the phenotypic space.

    """
    def __init__(self, maximize=True):
        """
        Create a MAX ONES problem with individuals that have bit vectors of
        size `length`
        """
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        >>> p = MaxOnes()
        >>> ind = core.Individual([0, 0, 1, 1, 0, 1, 0, 1, 1],
        ...                       decoder=core.IdentityDecoder(),
        ...                       problem=p)
        >>> p.evaluate(ind.decode())
        5
        """
        return phenome.count(1)
